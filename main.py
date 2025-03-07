import streamlit as st
import shap
import lime
import lime.lime_tabular
import joblib
import pandas as pd
import matplotlib.pyplot as plt

@st.fragment
def para_input(model, explainer, explainer2, ct):
    st.header("请在下方输入相应指标👇", anchor=False)
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN', 
                    'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road', 
                    'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']
    # ... [保持原有滑动条代码不变] ...

@st.fragment
def main(model, explainer, explainer2):
    # 添加简介部分
    st.title("四川云南省森林火灾预测系统")
    
    st.header("一. 关于模型")
    st.write("""
    模型的预测结果显示，ROC曲线下的面积（AUC）为0.962，表明该模型具有良好的预测性能。
    该模型使用LightGBM算法构建，综合考虑了气象、地形、人文等18个特征因素。
    """)
    
    st.header("二. 实时预测")
    st.write("预测结果显示：")
    
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN', 
                    'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road', 
                    'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']
    
    if True:
        fire_type = model.predict(st.session_state["features"])
        predicted_proba = model.predict_proba(st.session_state["features"])[0]
        types = ["不发生火灾", "发生火灾"]
        
        # 预测结果展示
        st.success(f'预测结果为：{types[fire_type[0]]}，概率为{round(predicted_proba[fire_type[0]], 2)}。')

        # SHAP和LIME解释部分
        st.header("三. SHAP和LIME局部解释")
        
        # 使用两列布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("SHAP解释")
            shap_values = explainer.shap_values(st.session_state["features"])
            exp = shap.Explanation(shap_values, explainer.expected_value, 
                                   st.session_state["features"], 
                                   feature_names=st.session_state["features"].columns)
            fig1, _ = plt.subplots()
            shap.waterfall_plot(exp[0], max_display=11)
            st.pyplot(fig1)
            st.caption("""
            SHAP力图将每个特征的贡献值可视化为推动预测值增加（正值）或减少（负值）的作用力。
            基准值（base value）表示模型的平均预测值，最终值（f(x)）是当前预测值。
            """)

        with col2:
            st.subheader("LIME解释")
            exp2 = explainer2.explain_instance(
                data_row=st.session_state["features"].values[0],
                predict_fn=model.predict_proba,
                num_features=11
            )
            fig2 = exp2.as_pyplot_figure()
            st.pyplot(fig2)
            st.caption("""
            LIME解释展示特征对当前预测的局部影响。绿色特征表示推动预测"发生火灾"的因素，
            红色特征表示抑制"发生火灾"的因素。数值大小表示特征重要性程度。
            """)

if __name__ == "__main__":
    # ... [保持原有模型加载和初始化代码不变] ...
    
    # 调整容器布局
    ct = st.container()
    with ct:  # 将简介内容放在主容器顶部
        main(model, explainer, explainer2)
    
    with st.sidebar:
        para_input(model, explainer, explainer2, ct)
