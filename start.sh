python main.py --simulation_setup_data "data/input/simulation_setup_data.json"  --candidate_csv "data/input/resume_samples.csv"


## getting evaluation data. PS: used the resume csv used in the simulation for this step.
python metrics/evaluation.py output_files/20240802_095722 data/input/resume_samples.csv




## Getting Bias data
#python metrics/bias.py /path/to/your/directory [--how {single,all}] [--index INDEX]

python metrics/bias.py output_files/20240802_095722 --how all