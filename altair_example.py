import altair as alt
from vega_datasets import data

print ('Loading Altair renderer')

# for the notebook only (not for JupyterLab) run this command once per session
alt.renderers.enable('notebook')


def example_basic():
  iris = data.iris()

  chart = alt.Chart(iris).mark_point().encode(
    x='petalLength',
    y='petalWidth',
    color='species',
  ).configure(
    background='white',
  )

  display(chart)


def example_geo():
  states = alt.topo_feature(data.us_10m.url, 'states')
  source = data.population_engineers_hurricanes.url
  variable_list = ['population', 'engineers', 'hurricanes']

  chart = alt.Chart(states).mark_geoshape().encode(
    alt.Color(alt.repeat('row'), type='quantitative')
  ).transform_lookup(
    lookup='id',
    from_=alt.LookupData(source, 'id', variable_list)
  ).properties(
    width=500,
    height=300
  ).project(
    type='albersUsa'
  ).repeat(
    row=variable_list
  ).resolve_scale(
    color='independent'
  ).configure(
    background='white',
  )


  display(chart)
