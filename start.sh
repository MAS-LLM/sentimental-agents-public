python3 main.py --simulation_setup_data "data/input/simulation_setup_data.json"  --candidate_csv "data/input/1sample.csv"


## getting evaluation data. PS: used the resume csv used in the simulation for this step.
python3 metrics/evaluation.py output_files/20240811_093714 data/input/samples.csv




## Getting Bias data
#python metrics/bias.py /path/to/your/directory [--how {single,all}] [--index INDEX]

python3 metrics/bias.py output_files/20240811_093714 --how all

python metrics/decision_making.py output_files/20240802_095722