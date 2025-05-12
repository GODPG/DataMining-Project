import random
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_original_vs_reconstructed(y_test, Xt_test, recon_t, feature=1, num_each=2):
    """绘制原始数据与重构数据的对比图"""
    # 随机选窗口
    norm_idx = np.where(y_test == 0)[0]
    anom_idx = np.where(y_test == 1)[0]
    sel_norm = random.sample(list(norm_idx), min(num_each, len(norm_idx)))
    sel_anom = random.sample(list(anom_idx), min(num_each, len(anom_idx)))
    sel = sel_norm + sel_anom

    for i in sel:
        # 将 ts_o 和 ts_r 转换为 NumPy 数组之前，先移动到 CPU
        ts_o = Xt_test[i, :, feature].cpu().numpy()
        ts_r = recon_t[i, :, feature].cpu().numpy()
        t = np.arange(len(ts_o))

        # 计算差值并找最大的 5 个索引
        diff = np.abs(ts_o - ts_r)
        # 取差值最大的 5 个位置
        idx5 = np.argsort(diff)[-5:]  # 或者 np.argpartition(diff, -5)[-5:]

        fig = go.Figure()

        # 原始 & 重构 折线
        fig.add_trace(go.Scattergl(
            x=t, y=ts_o, mode='lines',
            name='Original',
            line=dict(color='blue', width=3)
        ))
        fig.add_trace(go.Scattergl(
            x=t, y=ts_r, mode='lines',
            name='Reconstructed',
            line=dict(color='red', width=3)
        ))

        # 在五个最大差值处画虚线并标注差值
        for j in idx5:
            fig.add_shape(
                type='line',
                x0=t[j], y0=ts_o[j],
                x1=t[j], y1=ts_r[j],
                line=dict(color='gray', width=2, dash='dash'),
            )
            mid_y = (ts_o[j] + ts_r[j]) / 2
            fig.add_annotation(
                x=t[j], y=mid_y,
                text=f"{diff[j]:.3f}",
                showarrow=False,
                yanchor='bottom',
                font=dict(size=16, color='gray')
            )
            # 高亮这两个点
            fig.add_trace(go.Scattergl(
                x=[t[j], t[j]],
                y=[ts_o[j], ts_r[j]],
                mode='markers',
                marker=dict(color='gray', size=10),
                showlegend=False,
                hoverinfo='skip'
            ))

        # 布局
        fig.update_layout(
            title=f'Window {i} ({"Normal" if y_test[i] == 0 else "Anomaly"})',
            title_font=dict(size=24),
            font=dict(family="Arial", size=16),
            xaxis=dict(
                title="Time step", title_font=dict(size=18),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title="Value", title_font=dict(size=18),
                tickfont=dict(size=14)
            ),
            legend=dict(
                font=dict(size=16),
                y=0.95, x=0.05
            ),
            margin=dict(l=50, r=30, t=80, b=50),
            template='plotly_white',
            hovermode='x unified'
        )

        fig.show()

def plot_roc_curve(y_test, mse_t):
    """绘制 ROC 曲线"""
    fpr, tpr, _ = roc_curve(y_test, mse_t)
    roc_auc = auc(fpr, tpr)

    # 创建 Figure
    fig = go.Figure()

    # ROC 曲线主线
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=3),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))

    # 对角参考线
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip'
    ))

    # 布局美化
    fig.update_layout(
        title='ROC Curve',
        title_font=dict(size=24),
        xaxis=dict(
            title='False Positive Rate',
            title_font=dict(size=18),
            tickfont=dict(size=14),
            range=[0, 1]
        ),
        yaxis=dict(
            title='True Positive Rate',
            title_font=dict(size=18),
            tickfont=dict(size=14),
            range=[0, 1]
        ),
        legend=dict(font=dict(size=16), y=0.05, x=0.95, xanchor='right', yanchor='bottom'),
        template='plotly_white',
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode='x unified'
    )

    fig.show()

def plot_precision_recall_curve(y_test, mse_t):
    """绘制 Precision-Recall 曲线"""
    precision, recall, _ = precision_recall_curve(y_test, mse_t)
    ap = average_precision_score(y_test, mse_t)

    # 绘图
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'AP = {ap:.3f}',
            line=dict(shape='spline', smoothing=1.0, width=3, color='#0072B2')
        )
    )

    # 美化布局
    fig.update_layout(
        title=dict(
            text='Smoothed Precision-Recall Curve',
            x=0.5,
            font=dict(size=20, family='Arial', color='#333333')
        ),
        xaxis=dict(
            title='Recall',
            title_font=dict(size=16, color='#555555'),
            showgrid=True,
            gridcolor='#E5ECF6',
            zeroline=False,
            linecolor='#CCCCCC',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Precision',
            title_font=dict(size=16, color='#555555'),
            showgrid=True,
            gridcolor='#E5ECF6',
            zeroline=False,
            linecolor='#CCCCCC',
            tickfont=dict(size=12)
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='#CCCCCC',
            borderwidth=1,
            font=dict(size=12)
        ),
        paper_bgcolor='#F9F9F9',  # 整个画布背景
        plot_bgcolor='#FFFFFF',  # 图表内部背景
        margin=dict(l=60, r=40, t=80, b=60),
    )

    fig.show()

def plot_threshold_vs_f1(y_test, mse_t):
    """绘制阈值 vs F1 曲线"""
    from sklearn.metrics import f1_score
    qs = np.linspace(0.5, 0.99, 50)
    f1_scores = []
    for q in qs:
        thr = np.quantile(mse_t, q)
        y_pred_q = (mse_t > thr).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_q))

    df = pd.DataFrame({
        'Quantile': qs,
        'F1-score': f1_scores
    })

    # 用 plotly.express 画平滑曲线并美化
    fig = px.line(
        df,
        x='Quantile',
        y='F1-score',
        title='Quantile vs F1-score',
        labels={'Quantile': 'Quantile (q)', 'F1-score': 'F1-score'},
        line_shape='spline',  # 样条曲线
        render_mode='svg',  # 矢量渲染更平滑
        markers=True  # 打点
    )

    # 更新曲线样式
    fig.update_traces(
        line=dict(width=3, dash='solid', color='#E64A19'),  # 加粗、实线、橙红色
        marker=dict(size=6, color='#E64A19', line=dict(width=1, color='white'))
    )

    # 更新布局样式
    fig.update_layout(
        title=dict(
            text='Quantile vs F1-score',
            x=0.5,  # 居中
            font=dict(size=22, family='Arial', color='#333333')
        ),
        xaxis=dict(
            title='Quantile (q)',
            title_font=dict(size=16, color='#555555'),
            tickmode='linear',
            tick0=0.5,
            dtick=0.1,
            showgrid=True,
            gridcolor='#ECEFF1',
            zeroline=False,
            linecolor='#78909C',
            ticks='outside',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='F1-score',
            title_font=dict(size=16, color='#555555'),
            tickformat='.2f',
            showgrid=True,
            gridcolor='#ECEFF1',
            zeroline=False,
            linecolor='#78909C',
            ticks='outside',
            tickfont=dict(size=12),
            range=[0, 1]
        ),
        paper_bgcolor='#FAFAFA',  # 整个画布背景
        plot_bgcolor='#FFFFFF',  # 绘图区背景
        margin=dict(l=60, r=40, t=80, b=60),
    )

    fig.show()

def plot_mse_vs_timestamp(mse_t, y_test):
    # Create timestamps
    timestamps = np.arange(len(mse_t))

    # Create Plotly figure
    fig = go.Figure(data=[go.Scatter(
        x=timestamps,
        y=mse_t,
        mode='markers',
        marker=dict(
            color=y_test,
            colorscale='Viridis',
            showscale=True,
            size=8,
            line=dict(width=0.5, color='Black')
        )
    )])

    # Customize layout
    fig.update_layout(
        title=dict(
            text='MSE vs. Timestamp',
            x=0.5,
            font=dict(size=22, family='Arial', color='#333333')
        ),
        xaxis_title=dict(
            text='Timestamp',
            font=dict(size=18, family='Arial', color='#555555')
        ),
        yaxis_title=dict(
            text='MSE',
            font=dict(size=18, family='Arial', color='#555555')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#DDDDDD',
            zeroline=False,
            linecolor='#999999',
            linewidth=1.2,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#DDDDDD',
            zeroline=False,
            linecolor='#999999',
            linewidth=1.2,
            tickfont=dict(size=14)
        ),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(240,240,240,0.8)',
        margin=dict(l=80, r=50, t=90, b=70),
        showlegend=False  # Hide legend
    )

    # Display the figure
    fig.show()
