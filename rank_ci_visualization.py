# Dash dependencies
import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table as table
from dash.dependencies import Input, Output, ClientsideFunction

# Data manipulation dependencies
import pandas as pd
import numpy as np

# Plotly dependencies
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Specify data location
data_path = './Dash Project/transit_time.csv'

# Import and clean data
df = pd.read_csv(data_path, skiprows=1)
df = df[['area', 'est_total', 'moe_total']]
df = df.sort_values(by='est_total', ascending=False)
df['rank'] = list(range(1, len(df)+1))

# Add in Bonferroni-corrected joint confidence intervals
# By default MOE is for alpha = 0.1 (z = 1.645).
# Applying Bonferroni correction for 51 states,
# we want alpha = 0.1/51 which corresponds to z = 3.1.
df['total_lb'] = df['est_total'] - (3.1/1.645)*df['moe_total']
df['total_ub'] = df['est_total'] + (3.1/1.645)*df['moe_total']
df['rank_lb']=None
df['rank_ub']=None

# Set debug status
ci_debug = False

# Find joint confidence region for ranks
for area in df.area:
    # Store area confidence interval
    area_k_ci = (float(df[df.area == area]['total_lb']), 
                 float(df[df.area == area]['total_ub']))

    # Find the length of LambdaL_k and LambdaR_k
    LambdaL_k_len = (area_k_ci[1] < df['total_lb']).sum()
    LambdaR_k_len = (area_k_ci[0] > df['total_ub']).sum()

    # Find the overlap and length of LambdaO_k
    overlap = np.maximum(0, np.minimum(area_k_ci[1], df['total_ub']) - np.maximum(area_k_ci[0], df['total_lb']))
    LambdaO_k_len = (overlap!=0).sum()-1

    # Debug Output
    if ci_debug==True:
        print('Area: ', area, '\n')
        print('Theta CI: ', area_k_ci, '\n')
        print('Rank: ', int(df[df.area == area]['rank']), '\n')
        print('Rank CI: ', (LambdaL_k_len + 1, LambdaL_k_len + LambdaO_k_len + 1))
        print('|LambdaL_k|: ', LambdaL_k_len, '\n')
        print('|LambdaR_k|: ', LambdaR_k_len, '\n')
        print('|LambdaO_k|: ', LambdaO_k_len, '\n')

    # Add rank intervals to df
    df.loc[df.area==area, ['rank_lb']] = LambdaL_k_len + 1
    df.loc[df.area==area, ['rank_ub']] = LambdaL_k_len + LambdaO_k_len + 1



# ------------------
# HELPER FUNCTIONS
# ------------------

def left_column(child_elements, width='25%'):

    lcol = html.Div(
        id="left-column",
        className="left column",
        children=child_elements,
        style={'width':str(width), 'display':'inline-block', 'align':'left'},
    )

    return(lcol)

def right_column(child_elements):

    rcol = html.Div(
        id="right-column",
        className="right column",
        children=child_elements,
        style={'align':'right', 'display':'inline-block'}
    )
    return(rcol)

def ranking_table(tab_data):
    """
    Draw a table of state rankings. Should highlight ranks within a confidence interval
    when a state is selected (clicked) or hovered over.
    """
    my_tab = table.DataTable(
        id='interactive-ranking-table',
        columns=[{"name":i, "id":i} for i in ['area', 'rank']],
        data=tab_data.to_dict('records'),
        row_selectable="multi",
        selected_rows=[],
        style_cell_conditional=[
            {
                'if':{'column_id':c}, 
                'textAlign':'left'
            } for c in ['area', 'rank']    
        ] + 
        [
            {'if':{'column_id':'area'}, 'width':'1%'},
            {'if':{'column_id':'rank'}, 'width':'1%'},
        ], 
        style_as_list_view=True,
    )
    return(my_tab)

def draw_heatmap(hm_data):
    hm_data = hm_data.reset_index(drop=True)
    hm_data['rank']
    hm_mtx = np.zeros((len(hm_data), len(hm_data)))

    for i in range(len(hm_data)):
        hm_mtx[i, int(hm_data['rank_lb'][i]) : int(hm_data['rank_ub'][i])] = 1
        hm_mtx[i, int(hm_data['rank'][i]-1)] = 1

    #fig = ff.create_annotated_heatmap(hm_mtx, 
    #    annotation_text=None, 
    #    colorscale='Greys')

    fig = go.Figure(
        data=[go.Heatmap(
            x=hm_data['rank'],
            y=hm_data['area'],
            z=hm_mtx,
            colorscale='Greys',
            colorbar=None,
        )],
        layout=go.Layout(
            #title = 'The thing',
            xaxis = dict(
                showgrid=True,
                zeroline=False,
                showline=True,
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=False,
                showline=True,
            ),
            autosize=True,
            height=1000,
            hovermode='closest',
        )
    )

    return fig

# ------------------
# DRAW APPLICATION
# ------------------

# Create dash app layout
app = dash.Dash(__name__)

# HTML Document Container
app.layout = html.Div(
    id="app-container",
    children=[
                # Banner
                html.Div(
                    id='banner',
                    className='banner',
                    children=["blah"]
                ),
                
                # Left column
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[ranking_table(df)],
                    style={'width':'25%', 
                    'display':'inline-block', 
                    'float':'left', 
                    'align':'left'}
                ),

                # Right column
                html.Div(
                    id="right-column",
                    className="eight columns",
                    children=[dcc.Graph(figure=draw_heatmap(df))],
                    style={'width':'75%', 
                    'display':'inline-block', 
                    'align':'left',
                    'float':'right'}
                )
                #left_column(ranking_table(df)),
                #right_column(dcc.Graph(figure=draw_heatmap(df)))
              ]
)

#app.layout = html.Div([
#    dcc.Graph(figure=draw_heatmap(df))
#])

if __name__ == '__main__':
    app.run_server(debug=True)