import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import lime
from PIL import Image


def main(model, explainer, explainer2):
    # 页面标题和简介
    st.title("四川云南省森林火灾预测", anchor=False)
    
    # 简介部分
    st.subheader("一、关于模型", anchor=False)
    st.write("模型的预测结果显示，ROC曲线下面积AUC为0.962，表明该模型具有良好的预测性能。")
    
    st.subheader("二、实时预测", anchor=False)
    if True:
        # 预测结果显示
        fire_type = model.predict(st.session_state["features"])
        predicted_proba = model.predict_proba(st.session_state["features"])[0]
        types = ["不发生火灾", "发生火灾"]
        st.write(f'预测结果为：{types[fire_type[0]]}，概率为{round(predicted_proba[fire_type[0]], 2)}。')

    # 解释部分
    st.subheader("三、SHAP和LIME局部解释", anchor=False)
    
    # SHAP解释部分
    st.markdown("#### SHAP解释")
    shap_values = explainer.shap_values(st.session_state["features"])
    exp = shap.Explanation(shap_values, explainer.expected_value, 
                          st.session_state["features"], 
                          feature_names=st.session_state["features"].columns)
    
    fig, _ = plt.subplots()
    shap.waterfall_plot(exp[0], max_display=11)
    st.pyplot(fig)
    st.write("上图显示了SHAP力图，可用于将每个变量的SHAP值可视化为一个'力'，它可以增加（正值）或减少（负值）相对于其基线的预测，用于对单个样本预测结果的解释。")

    # LIME解释部分
    st.markdown("#### LIME解释")
    exp2 = explainer2.explain_instance(
        data_row=st.session_state["features"].values[0],
        predict_fn=model.predict_proba,
        num_features=11
    )
    
    fig2 = exp2.as_pyplot_figure()
    st.pyplot(fig2)
    st.write("上图显示了LIME的局部解释图，右侧的变量（绿色）表示对火灾发生的预测为正影响，左侧的变量（红色）表示对火灾发生的预测为负影响，数值大小表示变量的重要性程度。")

if __name__ == "__main__":
    model = joblib.load('lgbml.pkl')
    explainer=shap.TreeExplainer(model)

    data1=pd.read_excel('./数据删.xls')
    columns_to_drop = ['LONGITUDE','LATITUDE','火点','TMX','TMN','GST']
    X = data1.drop(columns=columns_to_drop)
    X.rename(columns={'TEM':'Da_AVGTEM', 'TMN':'Da_MINTEM', 'TMX':'Da_MAXTEM', 'PRE':'Da_PRE', 
                      'WIN':'Da_AVGWIN', 'PRS':'Da_AVGPRS','GST':'Da_AVGGST','WINMAX':'Da_MAXWIN',
                      'GSTMAX':'Da_MAXGST','RHU':'Da_AVGRH','高程':'Elevation', '坡度':'Slope',
                      '坡向':'Aspect','铁路欧':'Dis_to_railway','公路欧':'Dis_to_road',
                      '平均人':'Den_pop','平均gdp':'GDP','居民欧':'Dis_to_sett','forest':'Forest',
                      'twi':'TWI'}, inplace=True)
        
    explainer2 = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=['No Fire', 'Fire'],
        mode='classification',
        random_state=42
    )
    if "features" not in st.session_state:
        st.session_state["features"] = {}
    
    ct = st.container()
    with st.sidebar:
        para_input(model, explainer, explainer2, ct)
    # main(model, explainer, explainer2)

