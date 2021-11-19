from simpsom import run_colors_example
from simpsom.cluster.quality_threshold import qt_test
from simpsom.cluster.density_peak import dp_test

if __name__ == "__main__":
    
    qt_test()
    dp_test()
    run_colors_example()
    run_colors_example(train_algo='online', learning_rate=0.1, epochs=1000)
