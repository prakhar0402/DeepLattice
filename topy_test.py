import topy

#filename = 'topy_test/topy/examples/mbb_beam/beam_2d_reci_10_iters.tpd'
filename = 'topy_test/topy/examples/dogleg/dogleg_3d_etaopt_gsf.tpd'

t = topy.Topology()
t.load_tpd_file(filename)
t.set_top_params()
#print(t.topydict)
topy.optimise(t, save=True, dir='./iterations')
print('Optimized!')
