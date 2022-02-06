import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

##########
#Imports
##########
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from evalml.model_understanding.prediction_explanations import explain_predictions

##########
#Streamlit setup
##########
st.set_page_config(page_title='Inteledge - Sales Prediction Simulator', page_icon="💡", layout="centered", initial_sidebar_state="auto", menu_items=None)

##########
#Functions for the predictions and for the page layout
##########
@st.cache(allow_output_mutation=True)
def get_pickles():
	_, logical_types = pickle.load(open('sample_en.pkl', 'rb'))
	pipeline, expected_value, pdp, pdp_relationship = pickle.load(open('model_en.pkl', 'rb'))
	
	return pipeline, logical_types, expected_value, pdp, pdp_relationship

@st.cache(allow_output_mutation=True)
def get_samples(target):
	# loading a dataset sample
	df = pickle.load(open('df_resampled_en.pkl', 'rb'))
	columns = [target] + df.drop(target, axis=1).columns.tolist()

	# loading the predictions
	df[target] = best_pipeline.predict(df.drop(target, axis=1))
	df = df.replace({target: {1: 'Approved', 0: 'Rejected'}})
	
	df_negados = df[df[target]=='Rejected'].tail(5).reset_index(drop=True)
	df_negados = df_negados[columns]

	df_aprovados = df[df[target]=='Approved'].tail(5).reset_index(drop=True)
	df_aprovados = df_aprovados[columns]
	
	return df, df_negados, df_aprovados

@st.cache(allow_output_mutation=True)
def get_global_explanations(pdp, pdp_relationship):
	figures_oneway = []
	figures_twoway = []
	
	# By state
	fig = px.bar(pdp['State'].sort_values(by='partial_dependence', ascending=False),
				x='feature_values', y='partial_dependence',
				color_discrete_sequence=[px.colors.qualitative.Plotly[3]],
				template='plotly_white',
				labels={
					"partial_dependence": "Change in the prediction",
					"feature_values": 'State'
				})
	figures_oneway.append((fig, 'Influence of the different states over the predictions'))
	
	# Por variáveis numéricas
	keys = ['# Registered', '# Exempt', '# Confirmed', 'Order Day', 'Order Month']
	for i in range(len(keys)):
		fig = px.line(pdp[keys[i]], x='feature_values', y='partial_dependence',
					color_discrete_sequence=[px.colors.qualitative.Plotly[i]], template='plotly_white',
					labels={
						"partial_dependence": "Change in the prediction",
						"feature_values": keys[i] if keys[i].startswith('#') else f"Value of {keys[i]}"
					})
		figures_oneway.append((fig, f'Influence of the different values of {keys[i]} over the predictions'))
	
	# Por 2 variáveis
	for key in [('# Registered', '# Confirmed'), ('# Registered', '# Exempt'), ('# Confirmed', '# Exempt')]:
		def get_values(df):
			y = df.columns
			x = df.index
			z = df.values
			return x, y, z
		
		labels = key
		try:
			x, y, z = get_values(pdp_relationship[key])
		except:
			x, y, z = get_values(pdp_relationship[(key[1], key[0])])
		    
		fig = go.Figure(data=go.Contour(z=z, x=x, y=y, line_smoothing=0.85))
		fig.update_layout(xaxis_title=labels[0], yaxis_title=labels[1])
		figures_twoway.append((fig, f'Influence of the values of {labels[0]} and {labels[1]} over the predictions'))
	
	return figures_oneway, figures_twoway

def plot_importances(best_pipeline, df):
	# predictions
	pred = max(0, int(best_pipeline.predict(df).values[0]))

	df_plot = explain_predictions(pipeline=best_pipeline, input_features=df.reset_index(drop=True),
							y=None, top_k_features=len(df.columns), indices_to_explain=[0],
							include_explainer_values=True, output_format='dataframe')

	df_plot = df_plot.sort_values(by='quantitative_explanation')
	df_plot['Sum'] = expected_value+df_plot['quantitative_explanation'].cumsum()
	df_plot = df_plot.sort_values(by='quantitative_explanation')
	df_plot['Influences to this result?'] = df_plot['quantitative_explanation']<0
	df_plot = df_plot.round(2)

	col_names = []
	for col in df_plot['feature_names'].values:
		if col == 'Order Date':
			col_names.append(f'{col}<br><em>({df[col].astype(str).values[0]})</em>')
		elif col == 'Company':
			company_name = df[col].values[0]
			company_name = company_name if len(company_name) < 25 else company_name[:22] + '...'
			col_names.append(f'{col}<br><em>({company_name})</em>')
		else:
			col_names.append(f'{col}<br><em>({df[col].values[0]})</em>')

	fig_xai = go.Figure(go.Waterfall(
		name='Prediction',
		base=0,
		orientation="h",
		y=['Initial prediction'] + col_names + ['Final prediction'],
		x=[expected_value] + df_plot['quantitative_explanation'].values.tolist() + [0],
		measure=['absolute'] + ['relative']*len(df_plot) + ['total'],
		text=[expected_value] + [f'{str(int(x))}' for x in df_plot['Sum'].values] + [pred],
		textposition = "outside",
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
	))

	fig_xai.update_xaxes(range=[max(0, df_plot['Sum'].min()*0.7),
								max(expected_value, pred*1.3, df_plot['Sum'].max()*1.3)])
    
	fig_xai.update_layout(
		title=f'Main influencers to this prediction:<br>(Prediction: <b>USD {max(0, int(pred))}</b>)',
		showlegend=False,
		width=420,
		template='plotly',
		height=900
	)

	return fig_xai, pred

##########
#Preparing the simulator
##########
# loading the predictive models
best_pipeline, logical_types, expected_value, pdp, pdp_relationship = get_pickles()

# loading dataset samples
target = 'Order Total (USD)'
df, df_negados, df_aprovados = get_samples(target)

##########
#Section 1 - History
##########
col1, _, _ = st.columns(3)
with col1:
	st.image('inteledge.png')

st.title('Sales Simulator')
st.markdown('Imagine the following scenario: you are a sales manager responsible for selling training and teaching sessions to other companies. These sessions could be for public or private companies. Or they could be sessions to small or large companies. Another possibility is to have many paid participants or exempt participants. Now, one thing is offering courses to a large company. Another is to ensure you will have many **participants**: after all, this is how you get your money.')
st.markdown('Now, how can we forecast future outcomes? How can we simulate **how much** you will actually get? This is why we developed an AI for sales forecasts. Think that it may also be applied to other sales types -- can you imagine other possibilities suitable for your business? Are you interested in doing something similar to your company? Contact us in @inteledge.app on [Instagram](https://instagram.com/inteledge.app) or on [LinkedIn](https://www.linkedin.com/company/inteledge/)!')
st.markdown('Also, check out [other analyses we did to another dataset](https://share.streamlit.io/wmonteiro92/sales-credit-approval-analysis-demo/main/exploration.py) e and [another AI algorithm we created for you to test as much as you want](https://share.streamlit.io/wmonteiro92/sales-credit-approval-xai-demo/main/predictions_xai.py)!')
##########
#Section 2 - Simulator
##########
st.header('Test yourself!')
st.markdown('See what would be the predictions for new sales opportunities. Test different configurations and see on the chart how the algorithm came to this decision for this test dataset. The predictions are updated in **real-time**.')
col1, col2 = st.columns(2)

with col1:
	# variables 
	company = st.selectbox('Company',
		tuple(np.sort(df['Company'].unique())))

	state = st.selectbox('State',
		tuple(np.sort(df['State'].unique())))
		 
	registered = st.slider('Number of registered employees',
		int(df['# Registered'].min()), int(df['# Registered'].max()),
		int(df['# Registered'].median()))
		 
	exempt = st.slider('Number of exempt employees',
		int(df['# Exempt'].min()), int(df['# Exempt'].max()),
		int(df['# Exempt'].median()))
		 
	confirmed = st.slider('Number of confirmed employees',
		int(df['# Confirmed'].min()), int(df['# Confirmed'].max()),
		int(df['# Confirmed'].median()))

	order_date = st.date_input('When did the order took place?',
		datetime.now(), datetime.strptime('2021-01-01', '%Y-%m-%d'), datetime.now())

	more_than_10k = st.checkbox('Does the company have more than 10 thousand employees?')

	private = st.checkbox('Is it a private company?')

	nationwide = st.checkbox('Is it a nationwide company?')

with col2:
	# inference
	df_inference = pd.DataFrame([[company, state, registered, exempt, confirmed,
		more_than_10k, private, nationwide, order_date.month,
		order_date.day, order_date.year]],
		columns=['Company', 'State', '# Registered', '# Exempt', '# Confirmed',
				'More than 10k employees?', 'Private company?', 'Nationwide company?',
				'Order Month', 'Order Day', 'Order Year'])
	df_inference = df_inference[logical_types.keys()]
	df_inference.ww.init()
	df_inference.ww.set_types(logical_types=logical_types)

	fig_xai, predicao = plot_importances(best_pipeline, df_inference)
	st.plotly_chart(fig_xai)

##########
#Section 3 - Influencers
##########
st.header('Getting to know the patterns learned by the algorithm')
st.write('From a Data Science standpoint, you must know **how** an algorithm came to a given prediction. Naturally, no algorithm will be 100% correct all the time. On the other hand, is it really learning the relationships between data? Could we have an error in data or something that did not make sense in it? See below some of the patterns found by the AI from a training dataset.')

figures_oneway, figures_twoway = get_global_explanations(pdp, pdp_relationship)

for fig in figures_oneway:
	st.subheader(fig[1])
	st.plotly_chart(fig[0], use_container_width=True)

st.write('It is also possible to verify the relationships between separate attributes. For example, is it really true that whenever the number of people registered increases and the number of people confirmed increases, the order value will always increase? And what will be the proportion of this increase? The charts below help us better understand that in simpler terms: the lighter the color, the more money we may receive. Check the **# Exempt**, for example: as the number of exempts increases, the order amount also increases. Sounds counter-intuitive, but perhaps there are other attributes taken into account, as well. Of course: other columns are always taken into account as well. Therefore, imagine these images as a tool to help you understand the relationships learned by the algorithm, but not in a way that it is a fixed rule. After all, the business world is not mathematics, right?')
for fig in figures_twoway:
	st.subheader(fig[1])
	st.plotly_chart(fig[0], use_container_width=True)
    
st.markdown('Follow us in [Instagram](https://instagram.com/inteledge.app) and in [LinkedIn](https://www.linkedin.com/company/inteledge/)!')
