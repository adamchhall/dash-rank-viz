# Dash dependencies
import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table as table
from dash.dependencies import Input, Output

# Data manipulation dependencies
import pandas as pd
import numpy as np

# Plotly dependencies
import plotly.graph_objs as go

# Import data csv
df = pd.read_csv('cleaned_transit_time.csv')

# Create empty heatmap matrix
hm_mtx = np.zeros((len(df), len(df)))

# Fill in all possible ranks as grey colored cells
for i in range(len(df)):
    hm_mtx[(int(df['rank_lb'][i]) - 1) : (int(df['rank_ub'][i])), i] = 0.5

# Fill in "point estimate" rank as black colored cells
np.fill_diagonal(hm_mtx, 1)

# Rank lists
northeast_state_ranks = df.loc[df.region == 'Northeast']['rank']
west_state_ranks = df.loc[df.region == 'West']['rank']
south_state_ranks = df.loc[df.region == 'South']['rank']
midwest_state_ranks = df.loc[df.region == 'Midwest']['rank']

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

    if col_indices == None or len(col_indices) == 0:
        fig.update_layout(plot_bgcolor='white', 
                          xaxis=dict(ticks='', showticklabels=False), 
                          yaxis=dict(ticks='', showticklabels=False))
    else:
        fig.update_layout(template='plotly_white', title="90% Confidence Intervals for Total Travel Time")

    return fig


def draw_heatmap(hm_data, hm_mtx, col_indices):

    # Save rank range
    yaxis_ranks = hm_data['rank']

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

    if col_indices == None or len(col_indices) == 0:
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
                        dcc.Tabs(id='nav-tabs', value='tab-2', children=[
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
                              dcc.Graph(figure=draw_heatmap(df, hm_mtx, []),
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
            dcc.Graph(figure=draw_heatmap(df, hm_mtx, selected_rows), 
                      config={'displayModeBar':False,
                              'staticPlot':True}),
            dcc.Graph(figure=draw_errbar(df, selected_rows),
                      config={'displayModeBar':False,
                              'staticPlot':True})]

# Run the app
app.run_server(host='0.0.0.0', port=8080, debug=False)