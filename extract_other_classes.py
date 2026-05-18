import pandas as pd
import numpy as np
import wfdb
import ast
import os

path = r'C:\Users\Darshan Naidu\OneDrive\Desktop\ug project-2\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

def run_extraction():
    try:
        # Load diagnostic mapping
        agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    cls = agg_df.loc[key].diagnostic_class
                    if pd.notna(cls):
                        tmp.append(cls)
            return list(set(tmp))

        # Load database
        Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        target_classes = {
            'MI': 'Class 2 (Myocardial Infarction)',
            'CD': 'Class 3 (Conduction Disturbance)',
            'STTC': 'Class 4 (ST/T Change)',
            'HYP': 'Class 5 (Hypertrophy)'
        }

        print("--- EXTRACTING ACTUAL PATIENT DATA FOR BOARD WORK ---")
        for tc, class_name in target_classes.items():
            # Find the first valid record containing this exact superclass
            match = Y[Y.diagnostic_superclass.apply(lambda x: tc in x)].iloc[0]
            
            filename = match.filename_lr
            full_file = os.path.join(path, filename)
            
            # Load raw binary data
            record_data, metadata = wfdb.rdsamp(full_file)
            lead_II_data = record_data[:, 1]
            
            # Locate Heartbeat / Major Spike
            peak_index = np.argmax(lead_II_data)
            
            # 5-point window extraction
            board_window = lead_II_data[peak_index - 2 : peak_index + 3]
            
            print(f"\n--- {class_name} ---")
            print(f"Extracted from Patient Record: {filename}")
            print(f"Peak Voltage: {lead_II_data[peak_index]:.3f} mV at index {peak_index}")
            print(f"x(t) = {list(np.round(board_window, 3))}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_extraction()
