# Dash dependencies
import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table as table
from dash.dependencies import Input, Output, State, ClientsideFunction

# Data manipulation dependencies
import pandas as pd
import numpy as np

# Plotly dependencies
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Specify data location
data_path = 'transit_time.csv'

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

    # Reset index
    df = df.reset_index(drop=True)

    # Add regions
    #df['region'] = 'Other'
    #for area in df['area']:
    #    if area in ['Washington', 'Oregon', 'California', 'Alaska', 'Hawaii']:
    #        df.loc[df['area'].isin()] = 'P'

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
    when a state is selected.
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

def draw_heatmap(hm_data, col_indices):
    
    # Create empty heatmap matrix
    hm_mtx = np.zeros((len(hm_data), len(hm_data)))

    # Save rank range
    yaxis_ranks = hm_data['rank']
    
    # Fill in all possible ranks as grey colored cells
    for i in range(len(hm_data)):
        hm_mtx[(int(hm_data['rank_lb'][i]) - 1) : (int(hm_data['rank_ub'][i])), i] = 0.5

    # Fill in "point estimate" rank as black colored cells
    np.fill_diagonal(hm_mtx, 1)

    # Subset the matrix and hm_data according to selected rows
    hm_data = hm_data.loc[hm_data['rank'].isin([i+1 for i in col_indices])]
    hm_mtx = hm_mtx[:, col_indices]

    # Create the figure to return
    fig = go.Figure(
        data=[go.Heatmap(
            x=hm_data['area'],
            y=yaxis_ranks,
            z=hm_mtx,
            colorscale='Greys',
            xgap = 2,
            ygap = 2,
        )],
        layout=go.Layout(
            xaxis = dict(
                showgrid=True,
                zeroline=False,
                showline=True,
                type='category',
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=False,
                showline=True,
                type='category',
            ),
            autosize=True,
            height=750,
            hovermode=False,
        )
    )

    # Reverse the y-axis order from the default
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['xaxis']['dtick'] = 1
    fig['layout']['yaxis']['dtick'] = 1
    fig.update_xaxes(automargin=True, tickangle=90)
    fig.update_traces(showscale=False)

    if col_indices is None or len(col_indices) == 0:
        fig['layout'].update(plot_bgcolor='white')

    # Return the figure
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
                    children=[
                        html.Button('All States', id='all-button', n_clicks=0), 
                        html.Button('No States', id='none-button', n_clicks=0),
                        ranking_table(df)
                        ],
                    style={'width':'25%', 
                    'display':'inline-block', 
                    'float':'left', 
                    'align':'left'}
                ),

                # Right column
                html.Div(
                    id="right-column",
                    className="eight columns",
                    children=[dcc.Graph(figure=draw_heatmap(df, []))],
                    style={'width':'75%', 
                    'display':'inline-block', 
                    'align':'left',
                    'float':'right'}
                )
              ]
)

@app.callback(
    Output('interactive-ranking-table', 'style_data_conditional'),
    [Input('interactive-ranking-table', 'selected_rows')]
)
def update_styles(selected_rows):
    return [{
        'if':{'row_index': i},
        'background_color':'#D2F3FF'
    } for i in selected_rows]

@app.callback(
    Output('right-column', 'children'),
    [Input('interactive-ranking-table', 'selected_rows')]
)
def update_heatmap(selected_rows):
    selected_rows.sort()
    return [dcc.Graph(figure=draw_heatmap(df, selected_rows))]

@app.callback(
    Output('interactive-ranking-table', 'selected_rows'),
    [Input('all-button', 'n_clicks'),],
    [State('interactive-ranking-table', 'derived_virtual_data')]
)
def select_all_rows(n_clicks, selected_rows):
    if selected_rows is None:
        return []
    else:
        return list(range(51))

#@app.callback(
#    Output('interactive-ranking-table', 'selected_rows'),
#    [Input('none-button', 'n_clicks'),],
#    [State('interactive-ranking-table', 'derived_virtual_data')]
#)
#def select_all_rows(n_clicks, selected_rows):
#    return []

if __name__ == '__main__':
    app.run_server(debug=True)