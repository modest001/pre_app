import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import lime
from PIL import Image


@st.fragment
def para_input(model, explainer, explainer2, ct):
    st.header("请在下方输入相应指标👇", anchor=False)
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN', 
                    'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road', 
                    'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']
    f1 = st.slider("日平均温度(Da_AVGTEM):", min_value=-17, max_value=40, value=-17, step=1)
    f2 = st.slider("日降雨量(Da_PRE):", min_value=0, max_value=99, value=0, step=1)
    f3 = st.slider("日平均相对湿度(Da_AVGRH):", min_value=0, max_value=100, value=0, step=1)
    f4 = st.slider("日平均风速(Da_AVGWIN):", min_value=0.0, max_value=9.3, value=0.0, step=0.1, format="%0.1f")
    f5 = st.slider("日平均气压(Da_AVGPRS):", min_value=602.0, max_value=1008.7, value=602.0, step=0.1, format="%0.1f")
    f6 = st.slider("日照时数(SSD):", min_value=0.0, max_value=13.5, value=0.0, step=0.1, format="%0.1f")

    f7 = st.slider("日最大风速(Da_MAXWIN):", min_value=0, max_value=25, value=0, step=1)
    f8 = st.slider("日最高地表气温(Da_MAXGST):", min_value=-2.9, max_value=78.6, value=-2.9, step=0.1, format="%0.1f")
    f9 = st.slider("海拔(Elevation):", min_value=1, max_value=7713, value=1, step=1)
    f10 = st.slider("坡度(Slope):", min_value=0.0, max_value=89.3, value=0.0, step=0.1, format="%0.1f")
    # f11 = st.slider("坡向/Aspect:", min_value=1, max_value=8, value=1, step=1)
    direction_dict = {"平面": 0, "北": 1, "东北": 2, "东": 3, "东南": 4, "南": 5, "西南": 6, "西": 7, "西北": 8}
    slope = st.selectbox("坡向(Aspect):", ["平面", "北", "东北", "东", "东南", "南", "西南", "西", "西北"])
    f11 = direction_dict[slope]

    f12 = st.slider("地形湿度指数(TWI):", min_value=-1.18, max_value=35.08, value=-1.18, step=0.01, format="%0.2f")
    
    f13 = st.slider("到最近铁路距离(Dis_to_road):", min_value=0, max_value=664042, value=0, step=1)
    f14 = st.slider("到最近道路距离(Dis_to_railway):", min_value=0.0, max_value=23136.3, value=0.0, step=0.1, format="%0.1f")
    f15 = st.slider("到最近居民点距离(Dis_to_sett):", min_value=0.0, max_value=25385.9, value=0.0, step=0.1, format="%0.1f")
    f16 = st.slider("人口密度(Den_pop):", min_value=0.688, max_value=15025.000, value=0.688, step=0.001, format="%0.3f")
    f17 = st.slider("人均GDP(GDP):", min_value=0.275, max_value=109917.000, value=0.275, step=0.001, format="%0.3f")
    # f18 = st.slider("Forest:", min_value=0, max_value=8, value=0, step=1)
    forest_dict = {"针叶林": 0, "针阔叶混交林": 1, "阔叶林": 2, "灌丛": 3,
                    "草丛": 4, "草甸": 5, "高山植被": 6, "栽培植被": 7, "其他": 8}
    forest = st.selectbox("植被类型(Forest):", ["针叶林", "针阔叶混交林", "阔叶林", "灌丛", "草丛", "草甸", "高山植被", "栽培植被", "其他"])
    f18 = forest_dict[forest]

    data = {'Da_AVGTEM': [f1], 'Da_PRE': [f2], 'Da_AVGRH': [f3], 'Da_AVGWIN': [f4], 'Da_AVGPRS': [f5], 'SSD': [f6], 'Da_MAXWIN': [f7], 
            'Da_MAXGST': [f8], 'Elevation': [f9], 'Slope': [f10], 'Aspect': [f11], 'TWI': [f12], 'Dis_to_railway': [f13], 
            'Dis_to_road': [f14], 'Dis_to_sett': [f15], 'Den_pop': [f16], 'GDP': [f17], 'Forest': [f18]}
    features = pd.DataFrame(data, columns=feature_names)
    st.session_state["features"] = features
    pre_button = st.button('预测', type='primary')
    if pre_button:
        with ct:
            main(model, explainer, explainer2)


@st.fragment
def main(model, explainer, explainer2):
    # ==================== 页面标题 ====================
    st.title("四川云南省森林火灾预测", anchor=False)
    
    # ==================== 模型简介部分 ====================
    st.header("一、关于模型", anchor=False)
    st.markdown("""
    - 基于LightGBM的机器学习模型
    - **ROC曲线下面积AUC为0.962**，表明模型具有优秀的预测性能
    - 使用SHAP和LIME双解释框架
    - 训练数据涵盖18个地理气候特征
    """)

    # ==================== 实时预测部分 ====================
    st.header("二、实时预测", anchor=False)
    
    # 确保特征数据已初始化
    if "features" not in st.session_state:
        st.error("请先在侧边栏输入特征数据")
        return

    # 执行预测
    fire_type = model.predict(st.session_state["features"])
    predicted_proba = model.predict_proba(st.session_state["features"])[0]
    types = ["不发生火灾", "发生火灾"]
    
    # 使用卡片式布局展示结果
    with st.container(border=True):
        col_result, col_prob = st.columns(2)
        with col_result:
            st.metric("预测结论", types[fire_type[0]])
        with col_prob:
            st.metric("置信概率", f"{predicted_proba[fire_type[0]]:.2%}")

    # ==================== 双解释器布局 ====================
    st.header("三、SHAP和LIME局部解释", anchor=False)
    
    # 创建并排的列布局
    col_shap, col_lime = st.columns(2, gap="large")

    # SHAP解释部分
    with col_shap:
        st.subheader("SHAP解释", anchor=False)
        with st.spinner("生成SHAP解释..."):
            shap_values = explainer.shap_values(st.session_state["features"])
            exp = shap.Explanation(
                shap_values,
                explainer.expected_value,
                st.session_state["features"],
                feature_names=st.session_state["features"].columns
            )
            fig1, _ = plt.subplots(figsize=(8, 6))
            shap.waterfall_plot(exp[0], max_display=11, show=False)
            st.pyplot(fig1)
        st.caption("""
        **解读说明**：  
        - 红色特征：提升火灾风险的主要因素  
        - 蓝色特征：降低火灾风险的保护因素  
        - 基准值：模型预测的平均基准风险值（f(x) = {:.2f}）
        """.format(explainer.expected_value))

    # LIME解释部分
    with col_lime:
        st.subheader("LIME解释", anchor=False)
        with st.spinner("生成LIME解释..."):
            exp2 = explainer2.explain_instance(
                data_row=st.session_state["features"].values[0],
                predict_fn=model.predict_proba,
                num_features=11
            )
            fig2 = exp2.as_pyplot_figure()
            plt.tight_layout()
            st.pyplot(fig2)
        st.caption("""
        **解读说明**：  
        - → 绿色特征：促进火灾发生的正相关因素  
        - ← 红色特征：抑制火灾发生的负相关因素  
        - 数值大小：表示特征影响程度  
        - 预测概率：本地模型拟合结果
        """)

if __name__ == "__main__":
    # 模型初始化
    model = joblib.load('lgbml.pkl')
    
    # SHAP解释器初始化
    explainer = shap.TreeExplainer(model)
    
    # LIME解释器初始化
    data1 = pd.read_excel('./数据删.xls')
    columns_to_drop = ['LONGITUDE','LATITUDE','火点','TMX','TMN','GST']
    X = data1.drop(columns=columns_to_drop)
    X.rename(columns={
        'TEM':'Da_AVGTEM', 'TMN':'Da_MINTEM', 'TMX':'Da_MAXTEM',
        'PRE':'Da_PRE', 'WIN':'Da_AVGWIN', 'PRS':'Da_AVGPRS',
        'GST':'Da_AVGGST','WINMAX':'Da_MAXWIN','GSTMAX':'Da_MAXGST',
        'RHU':'Da_AVGRH','高程':'Elevation', '坡度':'Slope',
        '坡向':'Aspect','铁路欧':'Dis_to_railway','公路欧':'Dis_to_road',
        '平均人':'Den_pop','平均gdp':'GDP','居民欧':'Dis_to_sett',
        'forest':'Forest','twi':'TWI'
    }, inplace=True)
    
    explainer2 = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=['No Fire', 'Fire'],
        mode='classification',
        verbose=True
    )

    # 初始化特征数据
    if "features" not in st.session_state:
        # 设置默认示例数据
        default_features = X.iloc[[0]].copy()
        st.session_state["features"] = default_features
    
    # 页面布局
    with st.container():
        main(model, explainer, explainer2)
    with st.sidebar:
        para_input(model, explainer, explainer2)  # 确保此函数正确初始化特征
