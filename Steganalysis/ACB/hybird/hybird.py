from sklearn import svm
import numpy as np

# times = [ "0.1","0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]#, "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
times = [ "1"]
# emrs = ["0",  "100"]
emrs = ["0","10",  "20","30",  "40","50",  "60","70",  "80","90",  "100"]
stegos = ["liu","huang","yan2"]
for stego in stegos:
    for time in times:
        for emr in emrs:
            pbp_file = "D:\\Vstego\\ACB\\npy\\PBP5-f_%ss_%s_%s.npy" % (time, emr, stego)
            cmsdpd_file = "D:\\Vstego\\ACB\\npy\\cmsdpd9_%ss_%s_%s.npy" % ( time, emr, stego)
            hybird_file = "D:\\Vstego\\ACB\\npy\\hybird_%ss_%s_%s.npy" % ( time, emr, stego)
            # pbp_file = "D:\\paper1Data\\NB\\%s\\npy/PBP_%ss_%s_%s.npy" % (stego, time, emr, stego)
            # cmsdpd9_file = "D:\\paper1Data\\NB\\%s\\npy/cmsdpd_%ss_%s_%s.npy" % (stego, time, emr, stego)
            # hybird_file = "D:\\paper1Data\\NB\\%s\\npy/hybird_%ss_%s_%s.npy" % (stego, time, emr, stego)
            print (stego, time, emr, hybird_file)
            pbp = np.load(pbp_file)
            cmsdpd9 = np.load(cmsdpd_file)
            hybird_data = np.hstack((pbp, cmsdpd9[:, 1:]))
            np.save(hybird_file, hybird_data)
