from snml import bernoulli, cbernoulli
from getData import change_economy_data_to_zero_one_data

"""
test real economy data
"""

names_except_gdp = ['CPI', 'CCPI', 'RS', 'HS', 'NHPI', 'CA', 'D.UE', 'D.Brent', 'D.WTI']


def test_economy():
    economy_data = change_economy_data_to_zero_one_data()
    gdp_data = economy_data['GDP']
    for name in names_except_gdp:
        factor = economy_data[name]
        delta_x_to_y = bernoulli(factor) - cbernoulli(factor, gdp_data)
        delta_y_to_x = bernoulli(gdp_data) - cbernoulli(gdp_data, factor)
        print "%s->%-25s = %.2f\t\t\t%-26s->%s = %.2f" % (
            'GDP', name, delta_x_to_y, name, 'GDP', delta_y_to_x)


test_economy()
