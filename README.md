# Plan: Student Innovation Lab & Idea Incubator

## Context
Building a new module for the ERP where students (Class 7-12) submit ideas, get peer votes & teacher feedback, form teams, go through incubation stages (Idea → Research → Prototype → Pitch), and compete in school-wide pitch events. No existing module supports student-initiated projects — this fills that gap.

---

## Phase 1: Backend Foundation — Django App + Idea Board

### 1.1 Create Django app
```bash
cd /Users/siddardha/k12-projects/erp-backend
python manage.py startapp innovation_lab
```

### 1.2 New file: `innovation_lab/models.py`
**16 models total:**

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `IdeaCategory` | Tech, Science, Social Impact, Creative Arts | `name`, `icon`, `color`, `is_active`, `order` |
| `Idea` | Core idea submission | `title`, `description`, `category` FK, `status` (draft/submitted/approved/incubating/archived/rejected), `submitted_by` FK, `session_year`/`acad_session`/`branch`/`grade` FKs, `upvote_count`, `comment_count`, `media_files` ArrayField, `tags` ArrayField |
| `IdeaVote` | Upvote tracking | `idea` FK, `user` FK, `unique_together` |
| `IdeaComment` | Threaded comments | `idea` FK, `user` FK, `content`, `parent` self-FK, `is_teacher_comment` |
| `InnovationTeam` | Team per idea | `name`, `idea` OneToOne, `max_members`(5), `min_members`(3), `is_recruiting`, `mentor` FK, `media` ArrayField |
| `TeamMember` | Team membership | `team` FK, `user` FK, `role` (leader/researcher/builder/presenter), `status` (invited/accepted/declined/removed), `unique_together` |
| `IncubationStage` | Pipeline stage per team | `team` FK, `stage` (idea/research/prototype/pitch), `status` (not_started/in_progress/review_pending/approved/needs_revision), `mentor_feedback`, `mentor_reviewed_by` FK |
| `StageStatusHistory` | Audit trail | `stage` FK, `old_status`, `new_status`, `changed_by` FK, `notes` |
| `StageSubmission` | Uploads per stage | `stage` FK, `submitted_by` FK, `submission_type` (document/photo/video/link/checkin), `title`, `description`, `media_files` ArrayField |
| `PitchEvent` | Pitch day event | `title`, `description`, `event_date`, `end_date`, `venue`, `status` (upcoming/ongoing/completed/cancelled), `session_year`/`branch`/`grades`, `max_teams`, `banner_image` |
| `PitchEventTeam` | Team registration | `event` FK, `team` FK, `presentation_order`, `unique_together` |
| `ScoringRubric` | Scoring criteria | `event` FK, `criteria_name`, `description`, `max_score`(10.0), `weight`(1.0), `order` |
| `JudgeAssignment` | Judge assignment | `event` FK, `judge` FK(User), `unique_together` |
| `PitchScore` | Per-criteria score | `event_team` FK, `judge` FK, `rubric` FK, `score`(Float), `feedback`, `unique_together` |
| `InnovationBadge` | Badge definitions | `name`, `badge_type` (idea_submitted/team_formed/stage_cleared/pitch_participated/winner/runner_up/best_innovation/peoples_choice), `icon_url` |
| `StudentBadge` | Awarded badges | `badge` FK, `student` FK(User), `idea`/`team`/`event` FKs (nullable), `certificate_url` |

All models follow existing conventions: `is_delete`, `created_at` (auto_now_add), `updated_at` (auto_now), `created_by`/`updated_by` FKs.

Pattern references:
- Voting → `feature_request/models.py` `FeatureRequestVote`
- Comments → `school_feed/models.py` `FeedComments`
- Teams → `project_management/models.py` `Project`/`ProjectMember`
- Status history → `feature_request/models.py` `FeatureRequestStatusHistory`
- Scoring → `assessment/models/user_response_models.py` `UserResponse`

### 1.3 New file: `innovation_lab/serializers.py`
Key serializers following `feature_request/serializers.py` pattern:
- `IdeaCategorySerializer`
- `IdeaListSerializer` (with `is_voted` SerializerMethodField, nested category)
- `IdeaDetailSerializer` (full detail with comments, team info)
- `IdeaCreateSerializer`
- `IdeaCommentSerializer` (self-referential for threads)
- `InnovationTeamSerializer` / `TeamMemberSerializer`
- `IncubationStageSerializer` / `StageSubmissionSerializer`
- `PitchEventListSerializer` / `PitchEventDetailSerializer` / `PitchEventCreateSerializer`
- `ScoringRubricSerializer` / `PitchScoreSerializer`
- `InnovationBadgeSerializer` / `StudentBadgeSerializer`
- Dashboard serializers: `MentorDashboardSerializer`, `ParentProjectViewSerializer`

### 1.4 New file: `innovation_lab/permissions.py`
```python
IsStudent          # user_level == 13
IsTeacherOrAbove   # user_level in [1, 2, 8, 10, 11]
IsAdminOrPrincipal # user_level in [1, 2, 8]
IsParent           # user_level == 14
IsTeamLeader       # TeamMember.role == 'leader'
IsAssignedMentor   # InnovationTeam.mentor == request.user
IsAssignedJudge    # JudgeAssignment exists
```

### 1.5 New file: `innovation_lab/views.py`
Using `success_response`/`error_response` pattern from `feature_request/views.py`:

| View | Method | Endpoint | Who |
|------|--------|----------|-----|
| `IdeaCategoryListView` | GET | `categories/` | All |
| `IdeaListCreateView` | GET/POST | `ideas/` | All / Students |
| `IdeaDetailView` | GET/PATCH/DELETE | `ideas/<pk>/` | All / Owner |
| `IdeaVoteToggleView` | POST | `ideas/<pk>/vote/` | Students + Teachers |
| `IdeaCommentListCreateView` | GET/POST | `ideas/<idea_id>/comments/` | All |
| `IdeaCommentDetailView` | PATCH/DELETE | `comments/<pk>/` | Owner |
| `TeamCreateView` | POST | `teams/` | Student (idea owner) |
| `TeamDetailView` | GET/PATCH | `teams/<pk>/` | Team members |
| `TeamMemberInviteView` | POST/PATCH | `teams/<team_id>/members/` | Leader |
| `TeamMemberRemoveView` | DELETE | `teams/<team_id>/members/<pk>/` | Leader |
| `TeamSearchView` | GET | `teams/search/` | Students |
| `IncubationPipelineView` | GET | `teams/<team_id>/pipeline/` | Team + Mentor |
| `StageDetailView` | GET/PATCH | `stages/<pk>/` | Team + Mentor |
| `StageSubmissionCreateView` | POST | `stages/<stage_id>/submissions/` | Team members |
| `StageSubmissionDeleteView` | DELETE | `submissions/<pk>/` | Owner |
| `MentorStageReviewView` | POST | `stages/<pk>/review/` | Mentor |
| `PitchEventListCreateView` | GET/POST | `events/` | All / Admin |
| `PitchEventDetailView` | GET/PATCH/DELETE | `events/<pk>/` | All / Admin |
| `PitchEventTeamRegisterView` | POST | `events/<event_id>/register/` | Team leader |
| `ScoringRubricCRUDView` | GET/POST/PATCH/DELETE | `events/<event_id>/rubrics/` | Admin |
| `JudgeAssignmentView` | POST/DELETE | `events/<event_id>/judges/` | Admin |
| `PitchScoreEntryView` | POST/PATCH | `events/<event_id>/scores/` | Judges |
| `PitchEventResultsView` | GET | `events/<event_id>/results/` | All |
| `BadgeListView` | GET | `badges/` | All |
| `StudentBadgeListView` | GET | `students/<student_id>/badges/` | All |
| `IdeaBoardStatsView` | GET | `stats/` | All |
| `MentorDashboardView` | GET | `mentor/dashboard/` | Teachers |
| `ParentInnovationView` | GET | `parent/projects/` | Parents |
| `StudentPortfolioView` | GET | `student/portfolio/` | Students |
| `UploadMediaView` | POST | `upload/` | Team members |

### 1.6 New file: `innovation_lab/urls.py`
Following `feature_request/urls.py` pattern with `app_name = 'innovation_lab'`.

### 1.7 New file: `innovation_lab/admin.py`
Register all models with `list_display`, `list_filter`, `search_fields`.

---

## Phase 2: Existing Files to Modify (Backend)

### 2.1 `erp-backend/erp_revamp/settings/base.py`
- Add `'innovation_lab'` to `INSTALLED_APPS`

### 2.2 `erp-backend/erp_revamp/urls.py` (line 78, before Google auth)
- Add: `path("innovation-lab/", include("innovation_lab.urls")),`

### 2.3 `erp-backend/school_feed/models.py` (line 17)
- Add `('8', 'Innovation Lab')` to `module_types` tuple — so ideas/events appear in school feed

### 2.4 Run migrations
```bash
python manage.py makemigrations innovation_lab
python manage.py migrate
```

---

## Phase 3: Frontend — New Page Module

### 3.1 New directory structure
```
src/pages/innovationLab/
├── index.js                    # Entry with Tabs (Dashboard/Ideas/Teams/Events/Badges)
├── Dashboard/
│   └── index.js                # Stats cards, recent ideas, active events, my teams
├── IdeaBoard/
│   ├── index.js                # Idea listing with category filter, sort by votes/newest
│   ├── SubmitIdea.js           # Form: title, description, category, media upload
│   └── IdeaDetail.js           # Full idea + vote button + threaded comments
├── TeamFormation/
│   ├── index.js                # Browse/search recruiting teams
│   ├── TeamDetail.js           # Team view: members, roles, pipeline status
│   └── InviteMembers.js        # Student search + invite modal
├── Pipeline/
│   ├── index.js                # Steps component showing 4 stages with status
│   ├── StageDetail.js          # Stage detail: submissions, checkins, mentor feedback
│   └── SubmissionUpload.js     # File upload using existing dnd-file-upload component
├── PitchEvents/
│   ├── index.js                # Event listing (upcoming/past)
│   ├── EventDetail.js          # Event detail: participating teams, schedule, results
│   ├── CreateEvent.js          # Admin form: title, date, venue, rubrics, judges
│   ├── JudgeScoring.js         # Per-team per-rubric scoring interface
│   └── Results.js              # Leaderboard with scores
├── Badges/
│   └── index.js                # Badge gallery grid
├── MentorDashboard/
│   └── index.js                # Assigned teams table, pending reviews, feedback
└── ParentView/
    └── index.js                # Child's projects (read-only), team, stage progress
```

### 3.2 New file: `src/services/innovationLab/api.js`
Direct axios calls (NOT Redux saga) — following `featureRequest` service pattern:
```javascript
import axiosInstance from 'config/axios'
const BASE = '/innovation-lab'

export const getIdeas = (params) => axiosInstance.get(`${BASE}/ideas/`, { params })
export const createIdea = (data) => axiosInstance.post(`${BASE}/ideas/`, data)
export const toggleVote = (id) => axiosInstance.post(`${BASE}/ideas/${id}/vote/`)
// ... ~30 API functions for all endpoints
```

### 3.3 Ant Design components used
- **Idea Board**: `Card`, `List`, `Tag` (categories), `Button` (vote with heart icon), `Input.Search`, `Select` (filters), `Avatar`
- **Idea Detail**: `Descriptions`, `Comment` (threaded), `Drawer` for comments
- **Submit Idea**: `Form`, `Input`, `Input.TextArea`, `Select`, `Upload` (media)
- **Team Formation**: `Card`, `Avatar.Group`, `Tag` (roles), `Badge`, `Modal` (invite)
- **Pipeline**: `Steps` (horizontal, 4 stages), `Card` per stage, `Progress`, `Timeline`
- **Stage Detail**: `Tabs`, `Upload` (reuse `src/components/dnd-file-upload/`), `List`, `Drawer`
- **Pitch Events**: `Card`, `Tag` (status), `Statistic` (countdown)
- **Judge Scoring**: `Form`, `Slider` or `Rate` per rubric, `Table`, `InputNumber`
- **Results**: `Table` (leaderboard), `Statistic`, `Medal` icons
- **Badges**: `Card.Grid`, `Badge`, image display
- **Mentor Dashboard**: `Tabs`, `Table`, `Badge` (pending count), `Drawer`
- **Parent View**: `Card`, `Steps` (progress), `Timeline`, read-only

---

## Phase 4: Existing Files to Modify (Frontend)

### 4.1 `src/router.js` — Add ~16 lazy-loaded routes
```javascript
// Innovation Lab
{ path: '/innovation-lab', Component: lazy(() => import('pages/innovationLab/Dashboard')), exact: true },
{ path: '/innovation-lab/ideas', Component: lazy(() => import('pages/innovationLab/IdeaBoard')), exact: true },
{ path: '/innovation-lab/ideas/submit', Component: lazy(() => import('pages/innovationLab/IdeaBoard/SubmitIdea')), exact: true },
{ path: '/innovation-lab/ideas/:id', Component: lazy(() => import('pages/innovationLab/IdeaBoard/IdeaDetail')), exact: true },
{ path: '/innovation-lab/teams', Component: lazy(() => import('pages/innovationLab/TeamFormation')), exact: true },
{ path: '/innovation-lab/teams/:id', Component: lazy(() => import('pages/innovationLab/TeamFormation/TeamDetail')), exact: true },
{ path: '/innovation-lab/pipeline/:teamId', Component: lazy(() => import('pages/innovationLab/Pipeline')), exact: true },
{ path: '/innovation-lab/pipeline/:teamId/stage/:stageId', Component: lazy(() => import('pages/innovationLab/Pipeline/StageDetail')), exact: true },
{ path: '/innovation-lab/events', Component: lazy(() => import('pages/innovationLab/PitchEvents')), exact: true },
{ path: '/innovation-lab/events/create', Component: lazy(() => import('pages/innovationLab/PitchEvents/CreateEvent')), exact: true },
{ path: '/innovation-lab/events/:id', Component: lazy(() => import('pages/innovationLab/PitchEvents/EventDetail')), exact: true },
{ path: '/innovation-lab/events/:id/scoring', Component: lazy(() => import('pages/innovationLab/PitchEvents/JudgeScoring')), exact: true },
{ path: '/innovation-lab/events/:id/results', Component: lazy(() => import('pages/innovationLab/PitchEvents/Results')), exact: true },
{ path: '/innovation-lab/badges', Component: lazy(() => import('pages/innovationLab/Badges')), exact: true },
{ path: '/innovation-lab/mentor', Component: lazy(() => import('pages/innovationLab/MentorDashboard')), exact: true },
{ path: '/innovation-lab/parent', Component: lazy(() => import('pages/innovationLab/ParentView')), exact: true },
```

### 4.2 `src/services/menu/index.js` — Add menu entry (after feature-request block ~line 351)
```javascript
{
  key: 'innovation-lab',
  title: 'Innovation Lab',
  icon: 'fa fa-flask',
  userLevel: [1, 2, 8, 10, 11, 13, 14],
  children: [
    { title: 'Dashboard', key: 'il-dashboard', url: '/innovation-lab', userLevel: [1, 2, 8, 10, 11, 13, 14] },
    { title: 'Idea Board', key: 'il-ideas', url: '/innovation-lab/ideas', userLevel: [1, 2, 8, 10, 11, 13] },
    { title: 'Submit Idea', key: 'il-submit', url: '/innovation-lab/ideas/submit', userLevel: [13] },
    { title: 'My Teams', key: 'il-teams', url: '/innovation-lab/teams', userLevel: [1, 2, 8, 10, 11, 13] },
    { title: 'Pitch Events', key: 'il-events', url: '/innovation-lab/events', userLevel: [1, 2, 8, 10, 11, 13, 14] },
    { title: 'Badges', key: 'il-badges', url: '/innovation-lab/badges', userLevel: [1, 2, 8, 11, 13] },
    { title: 'Mentor Dashboard', key: 'il-mentor', url: '/innovation-lab/mentor', userLevel: [1, 2, 8, 10, 11] },
    { title: "Child's Projects", key: 'il-parent', url: '/innovation-lab/parent', userLevel: [14] },
  ],
},
```

---

## Phase 5: Implementation Order

### Step 1 — Backend Models + Migrations
- Create `innovation_lab` app
- Write all 16 models in `models.py`
- Add to `INSTALLED_APPS` in `base.py`
- Add URL include in `urls.py`
- Run `makemigrations` + `migrate`
- Register models in `admin.py`

### Step 2 — Backend APIs (Idea Board)
- Write `permissions.py`
- Write serializers for Idea, IdeaVote, IdeaComment, IdeaCategory
- Write views for Idea CRUD, voting, commenting
- Write `urls.py` with idea board routes
- Test with Postman/curl

### Step 3 — Frontend Idea Board
- Create `src/services/innovationLab/api.js`
- Add routes in `router.js`
- Add menu item in `menu/index.js`
- Build: Dashboard → IdeaBoard (list) → SubmitIdea (form) → IdeaDetail (with voting + comments)

### Step 4 — Backend APIs (Teams + Pipeline)
- Write serializers for Team, TeamMember, IncubationStage, StageSubmission
- Write views for team CRUD, invite, pipeline, submissions, mentor review
- Add routes

### Step 5 — Frontend Teams + Pipeline
- Build: TeamFormation (browse/create) → TeamDetail → Pipeline (Steps) → StageDetail (uploads + mentor review)

### Step 6 — Backend APIs (Pitch Events + Scoring)
- Write serializers for PitchEvent, PitchEventTeam, ScoringRubric, JudgeAssignment, PitchScore
- Write views for event CRUD, team registration, judge scoring, results aggregation
- Add routes

### Step 7 — Frontend Pitch Events
- Build: PitchEvents (list) → CreateEvent (admin) → EventDetail → JudgeScoring → Results

### Step 8 — Backend APIs (Badges + Dashboards)
- Write serializers for InnovationBadge, StudentBadge
- Write badge auto-award logic
- Write mentor dashboard, parent view, student portfolio APIs
- Add `('8', 'Innovation Lab')` to school_feed module_types

### Step 9 — Frontend Badges + Dashboards
- Build: Badges gallery → MentorDashboard → ParentView

### Step 10 — Polish
- Add `select_related`/`prefetch_related` to querysets
- Add `db_index` on frequently filtered fields
- Seed data migration for IdeaCategory and InnovationBadge defaults
- End-to-end testing per role

---

## Files Summary

### New Files (Backend — 8 files)
| File | Description |
|------|-------------|
| `erp-backend/innovation_lab/__init__.py` | App init |
| `erp-backend/innovation_lab/apps.py` | App config |
| `erp-backend/innovation_lab/models.py` | 16 models |
| `erp-backend/innovation_lab/serializers.py` | ~20 serializers |
| `erp-backend/innovation_lab/views.py` | ~30 views |
| `erp-backend/innovation_lab/urls.py` | ~30 URL patterns |
| `erp-backend/innovation_lab/permissions.py` | 7 permission classes |
| `erp-backend/innovation_lab/admin.py` | Admin registrations |

### New Files (Frontend — ~20 files)
| File | Description |
|------|-------------|
| `erp-frontend/src/services/innovationLab/api.js` | ~30 API functions |
| `erp-frontend/src/pages/innovationLab/index.js` | Entry point |
| `erp-frontend/src/pages/innovationLab/Dashboard/index.js` | Dashboard |
| `erp-frontend/src/pages/innovationLab/IdeaBoard/index.js` | Idea listing |
| `erp-frontend/src/pages/innovationLab/IdeaBoard/SubmitIdea.js` | Submit form |
| `erp-frontend/src/pages/innovationLab/IdeaBoard/IdeaDetail.js` | Idea detail |
| `erp-frontend/src/pages/innovationLab/TeamFormation/index.js` | Team browse |
| `erp-frontend/src/pages/innovationLab/TeamFormation/TeamDetail.js` | Team detail |
| `erp-frontend/src/pages/innovationLab/TeamFormation/InviteMembers.js` | Invite modal |
| `erp-frontend/src/pages/innovationLab/Pipeline/index.js` | 4-stage pipeline |
| `erp-frontend/src/pages/innovationLab/Pipeline/StageDetail.js` | Stage detail |
| `erp-frontend/src/pages/innovationLab/Pipeline/SubmissionUpload.js` | Upload component |
| `erp-frontend/src/pages/innovationLab/PitchEvents/index.js` | Events list |
| `erp-frontend/src/pages/innovationLab/PitchEvents/EventDetail.js` | Event detail |
| `erp-frontend/src/pages/innovationLab/PitchEvents/CreateEvent.js` | Admin create |
| `erp-frontend/src/pages/innovationLab/PitchEvents/JudgeScoring.js` | Scoring UI |
| `erp-frontend/src/pages/innovationLab/PitchEvents/Results.js` | Leaderboard |
| `erp-frontend/src/pages/innovationLab/Badges/index.js` | Badge gallery |
| `erp-frontend/src/pages/innovationLab/MentorDashboard/index.js` | Mentor view |
| `erp-frontend/src/pages/innovationLab/ParentView/index.js` | Parent view |

### Modified Files (3 backend + 2 frontend)
| File | Change |
|------|--------|
| `erp-backend/erp_revamp/settings/base.py` | Add `'innovation_lab'` to INSTALLED_APPS |
| `erp-backend/erp_revamp/urls.py` | Add `path("innovation-lab/", include("innovation_lab.urls"))` at line 78 |
| `erp-backend/school_feed/models.py` | Add `('8', 'Innovation Lab')` to `module_types` at line 17 |
| `erp-frontend/src/router.js` | Add 16 lazy-loaded route entries |
| `erp-frontend/src/services/menu/index.js` | Add innovation-lab menu block after line 351 |

---

## Verification
1. Backend: `python manage.py makemigrations innovation_lab` — should generate migration
2. Backend: `python manage.py migrate` — should apply cleanly
3. Backend: `python manage.py runserver` — verify no import errors
4. Backend: Test each API group via curl/Postman (ideas → teams → pipeline → events → badges)
5. Frontend: `npm run dev` — verify no build errors
6. Frontend: Navigate to `/innovation-lab` — verify menu appears and dashboard loads
7. End-to-end: Student submits idea → Votes → Team formed → Stages progressed → Pitch event → Scoring → Badges awarded
8. Role testing: Verify student, teacher, admin, parent, judge each see correct views
