from scripts.load_data.load_existing import run
from scripts.load_data.load_single import start, load_orignal_data
from tqdm import tqdm
import scripts.evaluate as evaluate
import scripts.base as base
import time
import scripts.evaluate as evaluate
import traceback


if __name__ == "__main__":
    
    #evaluate.parameter_calculations.calculate_parameters(only_new = False)
    if True:
        data = run()
        for val in tqdm(data, "preprocessing"):
            try:
                val.preprocess()
                val.last_processed = time.time()
                val.new_length = len(val.data)
                val.save()
            except Exception as e:
                print(f"ERROR: on {val.original_name} - path: {val.original_path} - ERROR:\n{e}")
                traceback.print_exc()
                pass

    if False:  # load_single
        name = "661427502554-light-curves"
        #name = "NGC 4593"
        #name = "IC 4329A"
        #name = "NGC 3783"
        #name = "WPVS 007"
        #name = "IRAS 06205-2316"
        
        #data = start(name,new_name = "NGC 4151")
        data = load_orignal_data(name)
        data.preprocess()
        data.plot_before_after()
        data.save()

    
    if True: # evaluate all
        data = base.load_processed_data()
        evaluate.evaluate_all_preprocessed()
        #evaluate.parameter_calculations.calculate_parameters()
        
        
        
#! verändert val.preprocess etwas??