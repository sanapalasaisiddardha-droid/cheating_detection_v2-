"""
PDF Report Generator
Creates detailed reports using ReportLab.
"""

import os
from typing import Dict, List, Any
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)


class ReportGenerator:
    """Generates PDF reports from detection results."""

    def __init__(self, config: dict):
        """
        Initialize the report generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.styles = getSampleStyleSheet()

        # Add custom styles
        self.styles.add(ParagraphStyle(
            name='AlertTitle',
            parent=self.styles['Heading2'],
            textColor=colors.darkred,
            spaceAfter=10
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            textColor=colors.darkblue,
            spaceBefore=20,
            spaceAfter=10
        ))

    def _format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to MM:SS format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted string like "5:30"
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def generate(
        self,
        alerts: List[Dict[str, Any]],
        video_info: Dict[str, Any],
        spatial,
        output_path: str
    ):
        """
        Generate the PDF report.

        Args:
            alerts: List of alert dictionaries
            video_info: Video metadata
            spatial: SpatialAnalyzer instance
            output_path: Path to save PDF
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        story = []

        # Title Page
        story.extend(self._build_title_page(video_info, len(alerts)))

        # Executive Summary
        story.extend(self._build_executive_summary(alerts, video_info, spatial))

        # Alert Details
        if alerts:
            story.append(PageBreak())
            story.extend(self._build_alert_details(alerts))

        # Methodology Section
        story.append(PageBreak())
        story.extend(self._build_methodology())

        # Build PDF
        doc.build(story)

    def _build_title_page(
        self,
        video_info: Dict[str, Any],
        alert_count: int
    ) -> List:
        """Build the title page elements."""
        elements = []

        # Title
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(
            "Exam Cheating Detection Report",
            self.styles['Title']
        ))

        elements.append(Spacer(1, 0.5*inch))

        # Subtitle with alert count
        if alert_count > 0:
            subtitle = f"{alert_count} Potential Cheating Incident(s) Detected"
            color = "red"
        else:
            subtitle = "No Cheating Incidents Detected"
            color = "green"

        elements.append(Paragraph(
            f'<font color="{color}">{subtitle}</font>',
            self.styles['Heading2']
        ))

        elements.append(Spacer(1, 1*inch))

        # Video Information Table
        duration = video_info.get('duration', 0)
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        info_data = [
            ['Video Duration', f'{minutes}m {seconds}s'],
            ['Original Resolution', f"{video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}"],
            ['Frame Rate', f"{video_info.get('fps', 'N/A'):.1f} fps"],
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
        ]

        info_table = Table(info_data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))

        elements.append(info_table)
        elements.append(PageBreak())

        return elements

    def _build_executive_summary(
        self,
        alerts: List[Dict[str, Any]],
        video_info: Dict[str, Any],
        spatial
    ) -> List:
        """Build the executive summary section."""
        elements = []

        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))

        # Summary statistics
        total_students = len(spatial.seats)
        teacher_found = spatial.teacher_id is not None

        summary_text = f"""
        <b>Analysis Overview:</b><br/>
        - Total students identified: {total_students}<br/>
        - Teacher identified: {'Yes' if teacher_found else 'No'}<br/>
        - Potential cheating pairs detected: {len(alerts)}<br/>
        """

        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.25*inch))

        # Alert Summary Table
        if alerts:
            elements.append(Paragraph("Alert Summary", self.styles['Heading3']))

            table_data = [['Students', 'Time Periods', 'Confidence', 'Tier', 'Primary Behaviors']]

            for alert in alerts:
                labels = alert['seat_labels']
                students = f"{labels[0]} & {labels[1]}"
                confidence = f"{alert['confidence']:.1%}"
                tier = alert['tier']
                behaviors = ', '.join(alert['behaviors'][:2]) if alert['behaviors'] else 'N/A'

                # Format timestamps
                if alert['timestamps']:
                    time_periods = ', '.join([
                        f"{self._format_timestamp(t[0])}-{self._format_timestamp(t[1])}"
                        for t in alert['timestamps'][:3]  # Show first 3 periods
                    ])
                    if len(alert['timestamps']) > 3:
                        time_periods += '...'
                else:
                    time_periods = 'N/A'

                # Truncate behaviors if too long
                if len(behaviors) > 40:
                    behaviors = behaviors[:37] + '...'

                table_data.append([students, time_periods, confidence, tier, behaviors])

            summary_table = Table(table_data, colWidths=[1*inch, 1.3*inch, 0.8*inch, 0.7*inch, 2.4*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))

            elements.append(summary_table)
        else:
            elements.append(Paragraph(
                "<i>No suspicious activity was detected during the analysis period.</i>",
                self.styles['Normal']
            ))

        return elements

    def _build_alert_details(self, alerts: List[Dict[str, Any]]) -> List:
        """Build detailed alert sections."""
        elements = []

        elements.append(Paragraph("Detailed Alert Analysis", self.styles['SectionHeader']))

        for i, alert in enumerate(alerts):
            labels = alert['seat_labels']

            # Alert header
            header_text = f"Alert {i+1}: {labels[0]} & {labels[1]}"
            elements.append(Paragraph(header_text, self.styles['AlertTitle']))

            # Confidence and tier
            elements.append(Paragraph(
                f"<b>Confidence Score:</b> {alert['confidence']:.1%} ({alert['tier']})",
                self.styles['Normal']
            ))

            # Timestamps
            if alert['timestamps']:
                timestamps_str = ', '.join([
                    f"{self._format_timestamp(t[0])} - {self._format_timestamp(t[1])}"
                    for t in alert['timestamps'][:5]
                ])
                if len(alert['timestamps']) > 5:
                    timestamps_str += ' ...'
                elements.append(Paragraph(
                    f"<b>Suspicious Time Periods:</b> {timestamps_str}",
                    self.styles['Normal']
                ))

            # Behaviors
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph("<b>Detected Behaviors:</b>", self.styles['Normal']))

            for behavior in alert['behaviors']:
                elements.append(Paragraph(f"  - {behavior}", self.styles['Normal']))

            elements.append(Spacer(1, 0.25*inch))

            # Add separator between alerts
            if i < len(alerts) - 1:
                elements.append(Paragraph("_" * 70, self.styles['Normal']))
                elements.append(Spacer(1, 0.2*inch))

        return elements

    def _build_methodology(self) -> List:
        """Build the methodology section."""
        elements = []

        elements.append(Paragraph("Methodology", self.styles['SectionHeader']))

        methodology_text = """
        This analysis was conducted using a 7-stage computer vision pipeline:
        <br/><br/>
        <b>1. Video Preprocessing</b><br/>
        Frames extracted at 5 fps with CLAHE contrast enhancement in LAB color space.
        <br/><br/>
        <b>2. Person Detection & Tracking</b><br/>
        YOLOv8 object detection with BoT-SORT multi-object tracking for consistent
        student identification throughout the video.
        <br/><br/>
        <b>3. Pose Estimation</b><br/>
        YOLOv8-Pose model extracts 17 COCO body keypoints. Head orientation is
        computed using solvePnP with facial landmarks.
        <br/><br/>
        <b>4. Spatial Analysis</b><br/>
        Automatic seat assignment based on median positions during warm-up period.
        Neighbor graphs built using distance thresholds. Teachers identified by
        movement patterns and standing posture.
        <br/><br/>
        <b>5. Behavioral Feature Extraction</b><br/>
        Per-frame features include: head direction, mutual gaze detection, whispering
        posture, body lean, hand position relative to face and neighbors, paper
        sharing gestures.
        <br/><br/>
        <b>6. Temporal Classification</b><br/>
        Sliding window analysis (10-second windows, 2-second stride) with weighted
        scoring. Multi-criteria filter requires: minimum suspicious windows, confidence
        threshold, mutual behavior evidence, and absence of nearby teacher.
        <br/><br/>
        <b>7. Alert Generation</b><br/>
        Confidence tiers (Very High > 85%, High > 70%, Medium > 55%) with timestamped
        evidence and behavior summaries.
        <br/><br/>
        <b>Important Notes:</b><br/>
        - This system is designed to flag potential incidents for human review.<br/>
        - False positives may occur due to normal classroom interactions.<br/>
        - Results should be verified by reviewing the evidence frames and video.<br/>
        - Teacher proximity during flagged periods may indicate supervised activity.
        """

        elements.append(Paragraph(methodology_text, self.styles['Normal']))

        return elements
