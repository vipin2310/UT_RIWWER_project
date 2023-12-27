from vierlinden.data.loader import VierlindenDataProcessor
from vierlinden.model.model import NHitsTrainingWrapper
from vierlinden.config import model_output_path

batch_size = 32
num_workers = 18

def main(arg : str):
    dp = VierlindenDataProcessor()
    df = dp.load_processed_data()
    df = dp.prepare_for_target(df, arg)
    
    training_df, test_df = dp.split_data(df)
    
    nhits_wrapper = NHitsTrainingWrapper(training_df, arg, batch_size, num_workers)
    optimal_lr = nhits_wrapper.find_optimal_learningrate()
    
    print(f"Optimal learning rate for {arg}: {optimal_lr}")
    
    trainer = nhits_wrapper.train(optimal_lr)
    
    trainer.save_checkpoint(model_output_path / f"nhits_{arg}.ckpt")
    

if __name__ == "__main__":
    main("Kaiserstr_outflow [l/s]")
    # main("Kreuzweg_outflow [l/s]")