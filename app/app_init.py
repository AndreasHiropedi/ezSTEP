from dash import Dash

my_app = Dash(__name__)
server = my_app.server
my_app.config.suppress_callback_exceptions = True
