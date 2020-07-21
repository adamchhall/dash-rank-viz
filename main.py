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
    northeast_state_ranks = list(df.loc[df.area.isin(['Connecticut', 'Maine', 
    'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont', 'New Jersey',
    'New York', 'Pennsylvania'])]['rank'])

    midwest_state_ranks = list(df.loc[df.area.isin(['Indiana', 'Illinois', 'Michigan',
    'Ohio', 'Wisconsin', 'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska',
    'North Dakota', 'South Dakota'])]['rank'])

    west_state_ranks = list(df.loc[df.area.isin(['California', 'Washington', 'Arizona', 'Colorado',
    'Oregon', 'Utah', 'Nevada', 'New Mexico', 'Idaho', 'Montana', 'Wyoming', 'Alaska',
    'Hawaii'])]['rank'])

    south_state_ranks = list(df.loc[df.area.isin(['Delaware', 'District of Columbia',
    'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia',
    'West Virginia', 'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas',
    'Louisiana', 'Oklahoma', 'Texas'])]['rank'])


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
        columns=[{"name":"State", "id":'area'}, {"name":"Rank", "id":"rank"}],
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

def draw_errbar(eb_data, col_indices):

    draw_data = eb_data.loc[eb_data['rank'].isin([i+1 for i in col_indices])]

    fig = go.Figure(data=go.Scatter(
        x = draw_data['area'],
        y = draw_data['est_total'],
        error_y = dict(
            type = 'data',
            array = [(3.1/1.645)*i for i in list(draw_data['moe_total'])],
            visible = True
        )))

    fig['layout']['xaxis']['dtick'] = 1

    if col_indices is None or len(col_indices) == 0:
        fig.update_layout(plot_bgcolor='white', 
                          xaxis=dict(ticks='', showticklabels=False), 
                          yaxis=dict(ticks='', showticklabels=False))
    else:
        fig.update_layout(template='plotly_white', title="90% Confidence Intervals for Total Travel Time")

    return fig


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
    else:
        fig.update_layout(title='90% Confidence Intervals for State Ranking (Total Travel Time)')

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
                # Left column
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[
                        dcc.Tabs(id='nav-tabs', value='tab-1', children=[
                            dcc.Tab(label='Regions', value='tab-1', children=[
                                dcc.RadioItems(id='region-select', options=[
                                    {'label':'None', 'value':'no-states'},
                                    {'label':'All', 'value':'all-states'},
                                    {'label':'Northeast', 'value':'ne-states'},
                                    {'label':'Midwest', 'value':'mw-states'},
                                    {'label':'West', 'value':'wst-states'},
                                    {'label':'South', 'value':'sth-states'},
                                ], labelStyle={'display':'block'}, value='no-states')
                            ]),
                            dcc.Tab(label='States', value='tab-2', children=[
                                ranking_table(df)
                            ]),
                        ]),
                        html.Div(id='nav-tab-content')
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
                    children=[html.H1(
                                  id='banner',
                                  className='banner',
                                  children=["Transit Time Visualization Demo"],
                                  style={
                                      'textAlign':'center'
                                  }
                              ),
                              html.H3(
                                  id='subbanner',
                                  className='banner',
                                  children="Estimated Total Travel Time to Work (2016 ACS)",
                                  style={
                                      'textAlign':'center'
                                  }
                              ),
                              dcc.Graph(figure=draw_heatmap(df, []),
                                        config={'displayModeBar':False,
                                                'staticPlot':True}),
                              dcc.Graph(figure=draw_errbar(df, []),
                                        config={'displayModeBar':False,
                                                'staticPlot':True})],
                    style={'width':'75%', 
                    'display':'inline-block', 
                    'align':'left',
                    'float':'right'}
                )
              ]
)

@app.callback(
    Output('interactive-ranking-table', 'selected_rows'),
    [Input('region-select', 'value')]
)
def region_select(value):
    if value=='no-states':
        return []
    if value=='all-states':
        return list(range(51))
    if value=='ne-states':
        return [i-1 for i in northeast_state_ranks]
    if value=='mw-states':
        return [i-1 for i in midwest_state_ranks]
    if value=='wst-states':
        return [i-1 for i in west_state_ranks]
    if value=='sth-states':
        return [i-1 for i in south_state_ranks]

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
    return [html.H1(id='banner',
                    className='banner',
                    children=["Transit Time Visualization Demo"],
                    style={
                        'textAlign':'center'
                    }),
            html.H3(id='subbanner',
                    className='banner',
                    children="Estimated Total Travel Time to Work (2016 ACS)",
                    style={
                        'textAlign':'center'
                    }
                ),
            dcc.Graph(figure=draw_heatmap(df, selected_rows), 
                      config={'displayModeBar':False,
                              'staticPlot':True}),
            dcc.Graph(figure=draw_errbar(df, selected_rows),
                      config={'displayModeBar':False,
                              'staticPlot':True})]

app.config['suppress_callback_exceptions'] = True

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)