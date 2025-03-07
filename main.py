import streamlit as st
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import lime
from lime import lime_tabular

# 初始化全局组件
def init_session_state():
    if "features" not in st.session_state:
        # 使用合理的默认值初始化特征数据
        default_features = pd.DataFrame([[
            -17, 0, 0, 0.0, 602.0, 0.0,        # f1-f6
            0, -2.9, 1, 0.0, 0, -1.18,         # f7-f12
            0, 0.0, 0.0, 0.688, 0.275, 0       # f13-f18
        ]], columns=[
            'Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD',
            'Da_MAXWIN', 'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI',
            'Dis_to_railway', 'Dis_to_road', 'Dis_to_sett', 'Den_pop', 'GDP', 'Forest'
        ])
        st.session_state["features"] = default_features

def main_content(model, explainer, lime_explainer):
    """主内容区域"""
    # ===== 页面标题 =====
    st.title("四川云南省森林火灾预测", anchor=False)
    
    # ===== 模型简介部分 =====
    st.header("一、关于模型", anchor=False)
    st.markdown("""
    - **模型类型**: 基于LightGBM的机器学习模型
    - **性能指标**: ROC曲线下面积AUC为0.962
    - **特征数量**: 18个地理气候特征
    - **解释框架**: SHAP + LIME双解释系统
    """)
    
    # ===== 实时预测结果 =====
    st.header("二、实时预测", anchor=False)
    with st.container(border=True):
        try:
            fire_type = model.predict(st.session_state["features"])
            proba = model.predict_proba(st.session_state["features"])[0]
            result_type = ["不发生火灾", "发生火灾"][fire_type[0]]
            
            cols = st.columns(2)
            cols[0].metric("预测结果", result_type)
            cols[1].metric("置信概率", f"{proba[fire_type[0]]:.2%}")
        except Exception as e:
            st.error(f"预测失败: {str(e)}")
    
    # ===== 解释器部分 =====
    st.header("三、SHAP和LIME解释", anchor=False)
    
    # 双列布局
    shap_col, lime_col = st.columns(2, gap="large")
    
    # SHAP解释
    with shap_col:
        st.subheader("SHAP解释", anchor=False)
        try:
            shap_values = explainer.shap_values(st.session_state["features"])
            exp = shap.Explanation(
                shap_values[0],  # 注意调整索引适配二分类问题
                base_values=explainer.expected_value[0],
                data=st.session_state["features"].values,
                feature_names=st.session_state["features"].columns
            )
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(exp[0], max_display=11, show=False)
            st.pyplot(plt.gcf())
            st.caption("SHAP值表示各特征对预测结果的贡献方向（正/负）及强度，基准值为模型预测的平均值。")
        except Exception as e:
            st.error(f"SHAP解释生成失败: {str(e)}")
    
    # LIME解释
    with lime_col:
        st.subheader("LIME解释", anchor=False)
        try:
            exp = lime_explainer.explain_instance(
                st.session_state["features"].values[0],
                model.predict_proba,
                num_features=11
            )
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("LIME解释展示局部特征重要性，绿色表示正向影响，红色表示负向影响。")
        except Exception as e:
            st.error(f"LIME解释生成失败: {str(e)}")

def sidebar_input():
    """侧边栏输入组件"""
    st.sidebar.header("参数输入", anchor=False)
    
    # 滑动条配置
    inputs = {
        'Da_AVGTEM': (-17, 40, -17),
        'Da_PRE': (0, 99, 0),
        'Da_AVGRH': (0, 100, 0),
        'Da_AVGWIN': (0.0, 9.3, 0.0, 0.1),
        'Da_AVGPRS': (602.0, 1008.7, 602.0, 0.1),
        'SSD': (0.0, 13.5, 0.0, 0.1),
        'Da_MAXWIN': (0, 25, 0),
        'Da_MAXGST': (-2.9, 78.6, -2.9, 0.1),
        'Elevation': (1, 7713, 1),
        'Slope': (0.0, 89.3, 0.0, 0.1),
        'TWI': (-1.18, 35.08, -1.18, 0.01),
        'Dis_to_railway': (0, 664042, 0),
        'Dis_to_road': (0.0, 23136.3, 0.0, 0.1),
        'Dis_to_sett': (0.0, 25385.9, 0.0, 0.1),
        'Den_pop': (0.688, 15025.0, 0.688, 0.001),
        'GDP': (0.275, 109917.0, 0.275, 0.001)
    }
    
    # 动态生成滑动条
    features = {}
    for feature, params in inputs.items():
        if len(params) == 3:
            features[feature] = st.sidebar.slider(
                label=f"{feature}:",
                min_value=params[0],
                max_value=params[1],
                value=params[2],
                step=1
            )
        else:
            features[feature] = st.sidebar.slider(
                label=f"{feature}:",
                min_value=params[0],
                max_value=params[1],
                value=params[2],
                step=params[3],
                format="%.2f" if params[3] < 1 else None
            )
    
    # 分类特征处理
    features['Aspect'] = st.sidebar.selectbox(
        "坡向(Aspect):",
        options=["平面", "北", "东北", "东", "东南", "南", "西南", "西", "西北"],
        index=0
    )
    
    features['Forest'] = st.sidebar.selectbox(
        "植被类型(Forest):",
        options=["针叶林", "针阔叶混交林", "阔叶林", "灌丛", "草丛", "草甸", "高山植被", "栽培植被", "其他"],
        index=0
    )
    
    # 编码分类特征
    aspect_dict = {v:i for i,v in enumerate(["平面", "北", "东北", "东", "东南", "南", "西南", "西", "西北"])}
    forest_dict = {v:i for i,v in enumerate(["针叶林", "针阔叶混交林", "阔叶林", "灌丛", "草丛", "草甸", "高山植被", "栽培植被", "其他"])}
    
    # 更新特征数据
    new_features = pd.DataFrame([[
        features['Da_AVGTEM'], features['Da_PRE'], features['Da_AVGRH'],
        features['Da_AVGWIN'], features['Da_AVGPRS'], features['SSD'],
        features['Da_MAXWIN'], features['Da_MAXGST'], features['Elevation'],
        features['Slope'], aspect_dict[features['Aspect']], features['TWI'],
        features['Dis_to_railway'], features['Dis_to_road'], features['Dis_to_sett'],
        features['Den_pop'], features['GDP'], forest_dict[features['Forest']]
    ]], columns=[
        'Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD',
        'Da_MAXWIN', 'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI',
        'Dis_to_railway', 'Dis_to_road', 'Dis_to_sett', 'Den_pop', 'GDP', 'Forest'
    ])
    
    if st.sidebar.button("更新预测", type="primary"):
        st.session_state["features"] = new_features
        st.rerun()

# 主程序
if __name__ == "__main__":
    # 初始化会话状态
    init_session_state()
    
    try:
        # 加载模型
        model = joblib.load('lgbml.pkl')
        
        # 初始化SHAP解释器
        shap_explainer = shap.TreeExplainer(model)
        
        # 初始化LIME解释器
        data = pd.read_excel('./数据删.xls')
        X = data.drop(columns=['LONGITUDE','LATITUDE','火点','TMX','TMN','GST'])
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            class_names=['无火灾', '有火灾'],
            mode='classification',
            verbose=True
        )
        
        # 页面布局
        sidebar_input()  # 加载侧边栏
        main_content(model, shap_explainer, lime_explainer)  # 加载主内容
        
    except FileNotFoundError as e:
        st.error(f"文件加载失败: {str(e)}")
    except Exception as e:
        st.error(f"系统错误: {str(e)}")
