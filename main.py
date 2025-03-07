import streamlit as st
import shap
import lime
import lime.lime_tabular
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def para_input(model, explainer, explainer2, ct):
    # 左侧部分：输入滑块部分
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
    
    direction_dict = {"平面": 0, "北": 1, "东北": 2, "东": 3, "东南": 4, "南": 5, "西南": 6, "西": 7, "西北": 8}
    slope = st.selectbox("坡向(Aspect):", ["平面", "北", "东北", "东", "东南", "南", "西南", "西", "西北"])
    f11 = direction_dict[slope]

    f12 = st.slider("地形湿度指数(TWI):", min_value=-1.18, max_value=35.08, value=-1.18, step=0.01, format="%0.2f")
    f13 = st.slider("到最近铁路距离(Dis_to_road):", min_value=0, max_value=664042, value=0, step=1)
    f14 = st.slider("到最近道路距离(Dis_to_railway):", min_value=0.0, max_value=23136.3, value=0.0, step=0.1, format="%0.1f")
    f15 = st.slider("到最近居民点距离(Dis_to_sett):", min_value=0.0, max_value=25385.9, value=0.0, step=0.1, format="%0.1f")
    f16 = st.slider("人口密度(Den_pop):", min_value=0.688, max_value=15025.000, value=0.688, step=0.001, format="%0.3f")
    f17 = st.slider("人均GDP(GDP):", min_value=0.275, max_value=109917.000, value=0.275, step=0.001, format="%0.3f")
    
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


def main(model, explainer, explainer2):
    # 右侧简介部分
    st.sidebar.header("四川云南省森林火灾预测")
    st.sidebar.subheader("一.关于模型")
    st.sidebar.write("模型的预测结果显示，ROC曲线下的面积AUC为0.962，表明该模型具有良好的预测性能。")
    
    st.sidebar.subheader("二.实时预测")
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN', 
                    'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road', 
                    'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']

    fire_type = model.predict(st.session_state["features"])
    predicted_proba = model.predict_proba(st.session_state["features"])[0]
    types = ["不发生火灾", "发生火灾"]

    st.sidebar.subheader("三.SHAP和LIME局部解释")
    # 预测结果展示
    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("预测结果", anchor=False)
        st.write(f'预测结果为：{types[fire_type[0]]}，概率为{round(predicted_proba[fire_type[0]], 2)}。')
    
    # 只展示LIME图，不展示SHAP依赖图
    st.subheader("LIME局部解释", anchor=False)
    exp2 = explainer2.explain_instance(
        data_row=st.session_state["features"].values[0],
        predict_fn=model.predict_proba,
        num_features=11
    )
    
    fig2 = exp2.as_pyplot_figure()
    st.pyplot(fig2)

    # 显示LIME解释文字
    st.write("上图显示了LIME的局部解释图，右侧的变量(绿色)表示对火灾预测为正，左侧的变量(红色)表示对火灾预测为负，下面的数值与变量的重要性相对应。")


if __name__ == "__main__":
    model = joblib.load('lgbml.pkl')
    explainer = shap.TreeExplainer(model)

    data1 = pd.read_excel('./数据删.xls')
    columns_to_drop = ['LONGITUDE', 'LATITUDE', '火点', 'TMX', 'TMN', 'GST']
    X = data1.drop(columns=columns_to_drop)
    X.rename(columns={'TEM': 'Da_AVGTEM', 'TMN': 'Da_MINTEM', 'TMX': 'Da_MAXTEM', 'PRE': 'Da_PRE', 
                      'WIN': 'Da_AVGWIN', 'PRS': 'Da_AVGPRS', 'GST': 'Da_AVGGST', 'WINMAX': 'Da_MAXWIN',
                      'GSTMAX': 'Da_MAXGST', 'RHU': 'Da_AVGRH', '高程': 'Elevation', '坡度': 'Slope',
                      '坡向': 'Aspect', '铁路欧': 'Dis_to_railway', '公路欧': 'Dis_to_road',
                      '平均人': 'Den_pop', '平均gdp': 'GDP', '居民欧': 'Dis_to_sett', 'forest': 'Forest',
                      'twi': 'TWI'}, inplace=True)
    
    explainer2 = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=['No Fire', 'Fire'],
        mode='classification'
    )
    
    if "features" not in st.session_state:
        st.session_state["features"] = {}

    ct = st.container()
    with st.sidebar:
        para_input(model, explainer, explainer2, ct)
