import pandas as pd

file_suffix = '.csv'


def merge_route_link(path, **kwargs):
    links_file = path + kwargs['links_file'] + file_suffix
    routes_file = path + kwargs['routes_file'] + file_suffix

    ###############
    # Load routes #
    ###############
    routes = pd.read_csv(routes_file)
    # Expand link_seq to multiple rows
    link_seq = routes['link_seq'].str.split(',').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
    link_seq.name = 'link_id'
    routes = routes.drop('link_seq', axis=1).join(link_seq)

    ###############
    # Load routes #
    ###############
    links = pd.read_csv(links_file)
    # determine cross in or cross out
    links['cross_in'] = 0
    links['cross_out'] = 0
    # Iterators links
    for index, row in links.iterrows():
        if ',' in str(row['in_top']):
            links.loc[index, "cross_in"] = 1
        if ',' in str(row['out_top']):
            links.loc[index, "cross_out"] = 1

    ##########################
    # Merge routes and links #
    ##########################
    links['link_id'] = links['link_id'].astype(str)
    routes['link_id'] = routes['link_id'].astype(str)
    routes_links = pd.merge(routes, links, on='link_id')
    routes_links.drop(['in_top', 'out_top'], axis=1, inplace=True)

    # temp table
    temp_routes_links = pd.DataFrame()

    # Count the number of cross in and cross out
    cross_in_number = routes_links[
        ['intersection_id', 'tollgate_id', 'cross_in']
    ].groupby(['intersection_id', 'tollgate_id'])['cross_in'].sum().reset_index().rename(
        columns={'cross_in': 'cross_in_number'}
    )
    cross_out_number = routes_links[
        ['intersection_id', 'tollgate_id', 'cross_out']
    ].groupby(['intersection_id', 'tollgate_id'])['cross_out'].sum().reset_index().rename(
        columns={'cross_out': 'cross_out_number'}
    )
    temp_routes_links = pd.merge(cross_in_number, cross_out_number, on=['intersection_id', 'tollgate_id'])

    # Cal length
    route_length = routes_links[
        ['intersection_id', 'tollgate_id', 'length']
    ].groupby(['intersection_id', 'tollgate_id'])['length'].sum().reset_index()
    temp_routes_links = pd.merge(temp_routes_links, route_length, on=['intersection_id', 'tollgate_id'], how='left')

    # Cal links number
    links_number = routes_links[
        ['intersection_id', 'tollgate_id']
    ].groupby(['intersection_id', 'tollgate_id']).size().reset_index(name='link_count')
    temp_routes_links = pd.merge(temp_routes_links, links_number, on=['intersection_id', 'tollgate_id'], how='left')

    # Cal length and links number for each lane
    for i in routes_links['lanes'].unique():
        # lane's length
        temp_length = routes_links[routes_links['lanes'] == i][
            ['intersection_id', 'tollgate_id', 'length']
        ].groupby(['intersection_id', 'tollgate_id'])['length'].sum().reset_index().rename(
            columns={'length': str(i) + '_length'}
        )

        # lane's links number
        temp_links = routes_links[routes_links['lanes'] == i][
            ['intersection_id', 'tollgate_id']
        ].groupby(['intersection_id', 'tollgate_id']).size().reset_index(name=str(i) + '_count')

        temp_routes_links = temp_routes_links.merge(
            temp_length, on=['intersection_id', 'tollgate_id'], how='left'
        ).merge(temp_links, on=['intersection_id', 'tollgate_id'], how='left')

    temp_routes_links.fillna(0, inplace=True)
    return temp_routes_links


def main():
    training_path = '../../dataSets/training/'
    output_path = '../../dataSets/training/'

    route_link_file = {
        'links_file': 'links (table 3)',
        'routes_file': 'routes (table 4)',
    }
    route_link = merge_route_link(training_path, **route_link_file)
    route_link.to_csv(output_path + 'route_link.csv', index=False)


if __name__ == '__main__':
    main()
