from simpsom import run_colors_example
from simpsom.cluster.quality_threshold import qt_test
from simpsom.cluster.density_peak import dp_test

if __name__ == "__main__":
    
    qt_test()
    dp_test()
    run_colors_example(train_algo='online', epochs=100, early_stop='bmudiff')
    run_colors_example(train_algo='batch', epochs=50, early_stop='bmudiff')
