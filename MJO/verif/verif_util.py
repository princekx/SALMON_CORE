import numpy as np
import glob, os, sys
import copy
import pandas as pd
import datetime
from bokeh.models import ColumnDataSource, HoverTool, Select, Div
from bokeh.models import LabelSet
from bokeh.models.glyphs import Text
from bokeh.plotting import figure, output_file, save
import configparser
import json
import warnings

# Set the global warning filter to ignore all warnings
warnings.simplefilter("ignore")

class MJODisplay:
    def __init__(self, model, config_values):
        self.config_values = config_values
        self.num_prev_days = 201
        self.parent_dir = '/home/users/prince.xavier/MJO/SALMON/MJO'

        self.model = model
        # 40 days of anlysis to be written out with the forecasts
        self.nanalysis2write = 40

    def write_dates_json(self, date):
        if self.model == 'mogreps':
            plot_dir = self.config_values['mogreps_mjo_plot_ens']
        elif self.model == 'glosea':
            plot_dir = self.config_values['glosea_mjo_plot_ens']

        # Specify the output file
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        json_file = os.path.join(plot_dir, f'{self.model}_dates.json')


        print(json_file)
        # Add a new date to the list
        new_date = date.strftime('%Y%m%d')

        if not os.path.exists(json_file):
            with open(json_file, 'w') as jfile:
                json.dump([new_date], jfile)

        # Load existing dates from dates.json
        with open(json_file, 'r') as file:
            existing_dates = json.load(file)


        # Check if the value exists in the list
        if new_date not in existing_dates:
            # Append the value if it doesn't exist
            existing_dates.append(new_date)

        existing_dates.sort()

        # Save the updated list back to dates.json
        with open(json_file, 'w') as file:
            json.dump(existing_dates, file, indent=2)

        print(f"The {json_file} file has been updated. New date added: {new_date}")

    def make_ens_mean_df(self, rmm_file_names):
        # read one file
        df = pd.read_csv(rmm_file_names[0])
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.strftime('%Y-%m-%d')
        # Make dummy for ensemble mean
        ens_mean_df = df.copy()
        # Loop through each CSV file, read it into a DataFrame, and concatenate to the combined_df
        for col in ['rmm1', 'rmm2', 'amp']:
            ens_mean_df[col] = np.array([pd.read_csv(file)[col] for file in rmm_file_names]).mean(axis=0)
        # Phase has to be integer
        ens_mean_df['phase'] = [int(x) for x in
                                np.array([pd.read_csv(file)['phase'] for file in rmm_file_names]).mean(axis=0)]
        return ens_mean_df

    def make_WHplot_template(self, title=None):
        '''
        # Generates the axes and background for the data to be plot on
        #
        :param title:
        :return:
        '''
        # Set up plot
        hover = HoverTool(tooltips=[
            ("Date", "@date"),
            ("RMM1", "@rmm1"),
            ("RMM2", "@rmm2"),
            ("Phase", "@phase"),
            ("Amp", "@amp"),
            ("Label", "@label"),
            ("Member", "@member")
        ], mode='mouse')
        plot = figure(height=500, width=500, tools=["pan, reset, save, wheel_zoom, box_zoom", hover],
                      x_range=[-4, 4], y_range=[-4, 4])

        plot.title.text = title

        # Mark the 8 sectors
        x = 4
        y = 0.707107
        linewidth = 0.25
        plot.line([-x, -y], [-x, -y], line_width=0.5, line_alpha=0.6)
        plot.line([y, x], [y, x], line_width=0.5, line_alpha=0.6)
        plot.line([-x, -y], [x, y], line_width=0.5, line_alpha=0.6)
        plot.line([y, x], [-y, -x], line_width=0.5, line_alpha=0.6)
        plot.line([-x, -1], [0, 0], line_width=0.5, line_alpha=0.6)
        plot.line([1, x], [0, 0], line_width=0.5, line_alpha=0.6)
        plot.line([0, 0], [-x, -1], line_width=0.5, line_alpha=0.6)
        plot.line([0, 0], [1, x], line_width=0.5, line_alpha=0.6)

        xt, yt = 3., 1.5
        phase_marker_source = ColumnDataSource(data=dict(xt=[-xt, -yt, yt, xt, xt, yt, -yt, -xt],
                                                         yt=[-yt, -xt, -xt, -yt, yt, xt, xt, yt],
                                                         phase_labels=[str(i) for i in range(1, 9)]))
        labels = LabelSet(x='xt', y='yt', text='phase_labels', level='glyph',
                          x_offset=0, y_offset=0, source=phase_marker_source,
                          text_color='grey', text_font_size="30pt", text_alpha=0.25)

        plot.add_layout(labels)
        plot.circle([0], [0], radius=1, color="white", line_color='grey', alpha=0.6)

        phase_name_source = ColumnDataSource(
            dict(x=[0, 0], y=[-3.75, 3.], text=['Indian \n Ocean', 'Western \n Pacific']))
        glyph = Text(x="x", y="y", text="text", angle=0., text_color="grey", text_align='center', text_alpha=0.25)
        plot.add_glyph(phase_name_source, glyph)

        phase_name_source = ColumnDataSource(dict(x=[-3.], y=[0], text=['West. Hem\n Africa']))
        glyph = Text(x="x", y="y", text="text", angle=np.pi / 2., text_color="grey", text_align='center',
                     text_alpha=0.25)
        plot.add_glyph(phase_name_source, glyph)

        phase_name_source = ColumnDataSource(dict(x=[3.], y=[0], text=['Maritime\n continent']))
        glyph = Text(x="x", y="y", text="text", angle=-np.pi / 2., text_color="grey", text_align='center',
                     text_alpha=0.25)
        plot.add_glyph(phase_name_source, glyph)

        plot.xaxis[0].axis_label = 'RMM1'
        plot.yaxis[0].axis_label = 'RMM2'

        return plot

    def bokeh_rmm_plot(self, date, members, title_prefix='MODEL'):
        print(self.config_values, 'HERE!!!')
        if self.model == 'mogreps':
            print(self.config_values)
            rmms_archive_dir = os.path.join(self.config_values['mogreps_mjo_archive_dir'],
                                            f'{date.strftime("%Y%m%d")}')
        elif self.model == 'glosea':
            rmms_archive_dir = os.path.join(self.config_values['glosea_mjo_archive_dir'],
                                            f'{date.strftime("%Y%m%d")}')

        if not os.path.exists(rmms_archive_dir):
            os.makedirs(rmms_archive_dir)

        rmm_file_names = [os.path.join(rmms_archive_dir,
                                       f'createdPCs.15sn.{date.strftime("%Y%m%d")}.fcast.{mem}.txt')
                          for mem in members]
        existing_files = [file_name for file_name in rmm_file_names if os.path.exists(file_name)]
        print(f'{len(existing_files)} RMM files found. Trying Bokeh plot.')

        # Read a dummy to get the analysis
        df = pd.read_csv(existing_files[0])
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.strftime('%Y-%m-%d')

        df_analysis = df.loc[df['label'] == 'analysis']
        # call the template
        plot = self.make_WHplot_template(title=f'{title_prefix} MJO Forecasts {date.strftime("%Y-%m-%d")}')

        # Plot analysis in grey
        plot.line('rmm1', 'rmm2', source=df_analysis, name="analysis",
                      line_color='grey', line_width=5, line_alpha=0.8)
        plot.circle('rmm1', 'rmm2', source=df_analysis, name="analysis_dots",
                        color='grey', radius=0.05, alpha=0.8)

        # plot each member in blue
        for mem, rmm_file_name in enumerate(existing_files):
            df = pd.read_csv(rmm_file_name)
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.strftime('%Y-%m-%d')
            fcast_start_index = min(df.loc[df['label'] == 'forecast'].index)
            # connect the forecasts to the analysis
            df_forecast = df.iloc[fcast_start_index - 1:]

            # Add member info
            df_forecast['member'] = [mem for i in range(len(df_forecast))]
            plot.line('rmm1', 'rmm2', source=df_forecast, name="analysis", line_color='blue', line_width=2,
                          line_alpha=0.1)
            plot.circle('rmm1', 'rmm2', source=df_forecast, name="analysis_dots", color='blue', radius=0.02,
                            alpha=0.1)

        # Plot ensemble mean
        ens_mean_df = self.make_ens_mean_df(existing_files)

        fcast_start_index = min(ens_mean_df.loc[ens_mean_df['label'] == 'forecast'].index)

        # connect the forecasts to the analysis
        ens_mean_df = ens_mean_df.iloc[fcast_start_index - 1:]

        # Add member info
        ens_mean_df['member'] = ['ens_mean' for i in range(len(ens_mean_df))]

        plot.line('rmm1', 'rmm2', source=ens_mean_df, name="analysis", line_color='blue', line_width=5,
                      line_alpha=0.5)
        plot.circle('rmm1', 'rmm2', source=ens_mean_df, name="analysis_dots", color='blue', radius=0.05,
                        alpha=0.3)

        if self.model == 'mogreps':
            plot_dir = self.config_values['mogreps_mjo_plot_ens']
        elif self.model == 'glosea':
            plot_dir = self.config_values['glosea_mjo_plot_ens']

        # Specify the output file
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_file = os.path.join(plot_dir,
                                 f'{title_prefix}_{date.strftime("%Y%m%d")}.html')
        output_file(plot_file)
        save(plot)

        # Write the json file
        self.write_dates_json(date)

        print(f'Plotted {plot_file}')
        # Display the plot
        #show(plot)