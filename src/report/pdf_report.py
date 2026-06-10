import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, ListFlowable, ListItem
)
from reportlab.pdfgen import canvas

from src.models.bridge import Bridge
from src.models.test_event import TestEvent
from src.models.modal_params import ModalParams
from src.models.damage_index import DamageIndex, DamageType
from src.models.alert import Alert
from src.temperature_compensation.compensation import TemperatureModel


def _add_page_number(canvas_obj: canvas.Canvas, doc):
    canvas_obj.saveState()
    canvas_obj.setFont('Helvetica', 9)
    canvas_obj.drawRightString(
        A4[0] - 2 * cm, 
        2 * cm, 
        f"第 {doc.page} 页"
    )
    canvas_obj.restoreState()


def _create_figure_image(fig, width: float = 16 * cm, height: float = 8 * cm) -> str:
    tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.write_image(tmp_file.name, width=width * 37.795, height=height * 37.795, scale=2)
    tmp_file.close()
    return tmp_file.name


def generate_health_report(
    bridge: Bridge,
    test_event: TestEvent,
    modal_params: ModalParams,
    baseline_params: Optional[ModalParams] = None,
    damage_indices: Optional[Dict[DamageType, DamageIndex]] = None,
    temperature_models: Optional[Dict[int, TemperatureModel]] = None,
    trend_figures: Optional[List] = None,
    alerts: Optional[List[Alert]] = None,
    output_path: Optional[str] = None
) -> str:
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(test_event.data.columns[0]) if hasattr(test_event, 'data') else '.',
            f"健康评估报告_{bridge.name}_{test_event.metadata.collection_time.strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        alignment=1,
        spaceAfter=20
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.darkblue,
        spaceBefore=15,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    story = []
    
    story.append(Paragraph("桥梁结构健康评估报告", title_style))
    story.append(Spacer(1, 0.5 * cm))
    
    bridge_info = [
        ["桥梁名称", bridge.name],
        ["桥梁ID", bridge.id],
        ["描述", bridge.description or "-"],
        ["测点数量", str(len(bridge.sensors))],
        ["报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    
    bridge_table = Table(bridge_info, colWidths=[4 * cm, 11 * cm])
    bridge_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(bridge_table)
    story.append(Spacer(1, 0.5 * cm))
    
    story.append(Paragraph("一、本次测试概况", subtitle_style))
    test_info = [
        ["测试名称", test_event.name],
        ["采集时间", test_event.metadata.collection_time.strftime("%Y-%m-%d %H:%M:%S")],
        ["天气情况", test_event.metadata.weather],
        ["环境温度", f"{test_event.metadata.temperature:.1f}°C" if test_event.metadata.temperature else "-"],
        ["风速", f"{test_event.metadata.wind_speed:.1f} m/s" if test_event.metadata.wind_speed else "-"],
        ["交通状态", test_event.metadata.traffic_status],
        ["采样频率", f"{test_event.sampling_rate:.0f} Hz"],
        ["数据点数", str(len(test_event.data))],
        ["测试时长", f"{len(test_event.data) / test_event.sampling_rate:.1f} s"],
        ["操作人员", test_event.metadata.operator or "-"],
        ["备注", test_event.metadata.notes or "-"]
    ]
    test_table = Table(test_info, colWidths=[4 * cm, 11 * cm])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgreen),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(test_table)
    
    story.append(PageBreak())
    
    story.append(Paragraph("二、模态参数识别结果", subtitle_style))
    
    if modal_params.mode_shapes:
        freq_data = [["模态阶次", "频率 (Hz)", "阻尼比 (%)", "阻尼质量", "MAC值"]]
        for i, mode in enumerate(modal_params.mode_shapes):
            quality = "良好" if mode.damping_quality == "good" else "较差"
            mac = f"{mode.mac_value:.3f}" if mode.mac_value else "-"
            freq_data.append([
                f"第{i+1}阶",
                f"{mode.frequency:.4f}",
                f"{mode.damping_ratio * 100:.4f}",
                quality,
                mac
            ])
        
        freq_table = Table(freq_data, colWidths=[2.5 * cm, 3 * cm, 3 * cm, 2.5 * cm, 3 * cm])
        freq_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(freq_table)
        story.append(Spacer(1, 0.5 * cm))
    
    from src.visualization.mode_shape import create_multi_mode_comparison, create_mac_heatmap
    
    try:
        mode_fig = create_multi_mode_comparison(bridge, modal_params)
        mode_img_path = _create_figure_image(mode_fig, width=16 * cm, height=6 * cm)
        story.append(Image(mode_img_path, width=16 * cm, height=6 * cm))
        story.append(Spacer(1, 0.3 * cm))
        os.unlink(mode_img_path)
    except Exception as e:
        story.append(Paragraph(f"振型图生成失败: {str(e)}", body_style))
    
    if baseline_params is not None:
        story.append(Paragraph("与基准模态MAC矩阵", styles['Heading3']))
        try:
            mac_fig = create_mac_heatmap(
                baseline_params, modal_params,
                "基准测试", "本次测试"
            )
            mac_img_path = _create_figure_image(mac_fig, width=12 * cm, height=10 * cm)
            story.append(Image(mac_img_path, width=12 * cm, height=10 * cm))
            os.unlink(mac_img_path)
        except Exception as e:
            story.append(Paragraph(f"MAC矩阵图生成失败: {str(e)}", body_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("三、损伤指标分析", subtitle_style))
    
    if damage_indices:
        for dt, di in damage_indices.items():
            story.append(Paragraph(f"3.1 {dt.value}", styles['Heading3']))
            
            if len(di.values) > 0:
                idx_data = [["位置/模态", "指标值", "阈值", "状态"]]
                for j, (loc, val) in enumerate(zip(di.locations or [], di.values)):
                    status = "超限" if di.threshold and abs(val) > abs(di.threshold) else "正常"
                    color = colors.red if status == "超限" else colors.green
                    idx_data.append([
                        loc or f"测点{j+1}",
                        f"{val:.4f}",
                        f"{di.threshold:.4f}" if di.threshold else "-",
                        status
                    ])
                
                idx_table = Table(idx_data, colWidths=[4 * cm, 3.5 * cm, 3.5 * cm, 3 * cm])
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]
                for row_idx in range(1, len(idx_data)):
                    if idx_data[row_idx][3] == "超限":
                        table_style.append(('TEXTCOLOR', (-1, row_idx), (-1, row_idx), colors.red))
                    else:
                        table_style.append(('TEXTCOLOR', (-1, row_idx), (-1, row_idx), colors.green))
                
                idx_table.setStyle(TableStyle(table_style))
                story.append(idx_table)
                story.append(Spacer(1, 0.3 * cm))
    
    story.append(Paragraph("四、温度补偿说明", subtitle_style))
    if temperature_models:
        temp_info = [["模态阶次", "模型类型", "R²值", "补偿状态"]]
        for mode_idx, model in temperature_models.items():
            temp_info.append([
                f"第{mode_idx+1}阶",
                "线性回归" if model.model_type == 'linear' else "二次多项式",
                f"{model.r_squared:.4f}",
                "已补偿"
            ])
        temp_table = Table(temp_info, colWidths=[3 * cm, 4 * cm, 3 * cm, 3 * cm])
        temp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkorange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(temp_table)
    else:
        story.append(Paragraph("未进行温度补偿（温度数据不足或温度模型未建立）", body_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("五、长期趋势分析", subtitle_style))
    if trend_figures:
        for i, fig in enumerate(trend_figures[:3]):
            try:
                img_path = _create_figure_image(fig, width=16 * cm, height=7 * cm)
                story.append(Image(img_path, width=16 * cm, height=7 * cm))
                story.append(Spacer(1, 0.3 * cm))
                os.unlink(img_path)
            except Exception as e:
                story.append(Paragraph(f"趋势图{i+1}生成失败: {str(e)}", body_style))
    else:
        story.append(Paragraph("长期趋势数据不足", body_style))
    
    story.append(Paragraph("六、预警记录", subtitle_style))
    if alerts:
        alert_data = [["时间", "级别", "指标", "当前值", "阈值", "状态"]]
        for alert in alerts[:10]:
            level_color = colors.red if alert.level.value == 'red' else colors.orange if alert.level.value == 'yellow' else colors.blue
            level_text = "红色预警" if alert.level.value == 'red' else "黄色预警" if alert.level.value == 'yellow' else "信息"
            alert_data.append([
                alert.trigger_time.strftime("%Y-%m-%d %H:%M"),
                level_text,
                alert.metric,
                f"{alert.current_value:.4f}",
                f"{alert.threshold:.4f}",
                "已确认" if alert.acknowledged else "待处理"
            ])
        alert_table = Table(alert_data, colWidths=[3.2 * cm, 2 * cm, 4 * cm, 2.2 * cm, 2.2 * cm, 2 * cm])
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        alert_table.setStyle(TableStyle(table_style))
        story.append(alert_table)
    else:
        story.append(Paragraph("无预警记录", body_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("七、评估结论", subtitle_style))
    
    conclusion = []
    overall_status = "正常"
    risk_level = "低"
    
    if alerts:
        unacknowledged = [a for a in alerts if not a.acknowledged]
        if unacknowledged:
            has_red = any(a.level.value == 'red' for a in unacknowledged)
            has_yellow = any(a.level.value == 'yellow' for a in unacknowledged)
            if has_red:
                overall_status = "异常"
                risk_level = "高"
            elif has_yellow:
                overall_status = "注意"
                risk_level = "中"
    
    if damage_indices:
        for dt, di in damage_indices.items():
            if di.threshold is not None and len(di.values) > 0:
                if np.any(np.abs(di.values) > abs(di.threshold)):
                    if overall_status != "异常":
                        overall_status = "注意"
    
    status_color = colors.green if overall_status == "正常" else colors.orange if overall_status == "注意" else colors.red
    
    conclusion.append(Paragraph(f"<b>总体状态：<font color='{status_color.hexval()}'>{overall_status}</font></b>", body_style))
    conclusion.append(Paragraph(f"<b>风险等级：<font color='{status_color.hexval()}'>{risk_level}</font></b>", body_style))
    conclusion.append(Spacer(1, 0.2 * cm))
    
    if len(modal_params.mode_shapes) > 0:
        conclusion.append(Paragraph(
            f"本次测试共识别出 {len(modal_params.mode_shapes)} 阶模态参数，"
            f"基频为 {modal_params.mode_shapes[0].frequency:.4f} Hz。",
            body_style
        ))
    
    if baseline_params is not None and len(modal_params.mode_shapes) > 0 and len(baseline_params.mode_shapes) > 0:
        freq_change = (modal_params.mode_shapes[0].frequency - baseline_params.mode_shapes[0].frequency) / baseline_params.mode_shapes[0].frequency * 100
        conclusion.append(Paragraph(
            f"与基准测试相比，基频变化率为 {freq_change:+.2f}%。",
            body_style
        ))
    
    if overall_status == "正常":
        conclusion.append(Paragraph("桥梁结构状态良好，各项指标均在正常范围内。", body_style))
        conclusion.append(Paragraph("建议：继续按计划进行定期监测。", body_style))
    elif overall_status == "注意":
        conclusion.append(Paragraph("部分指标出现异常，需要关注。", body_style))
        conclusion.append(Paragraph("建议：加强监测频率，对异常指标进行重点关注，必要时进行现场检查。", body_style))
    else:
        conclusion.append(Paragraph("检测到严重异常，可能存在结构损伤风险！", body_style))
        conclusion.append(Paragraph("建议：立即进行全面的结构检查，限制桥梁通行，组织专家评估。", body_style))
    
    for item in conclusion:
        story.append(item)
    
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("报告审核人：________________", body_style))
    story.append(Paragraph("审核日期：________________", body_style))
    
    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    
    return output_path
