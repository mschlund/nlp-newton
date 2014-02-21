import numpy as np
import csv
from functools import partial



bank = "TueBaDZ"

seeds = [1271,49647,53316,25907,13023,898,30964,51253,54469,45047]


def main() :
  for unfold in [True, False] :
    for dim_height in ["D", "H"] :
      for VMarkov in [0,1] :
          speed = {}
          for seed in seeds:
            stat_filename = "speeds_" + (dim_height + "_" if unfold else '') + str(seed) + ('_VM'+str(VMarkov) if VMarkov>0 else '') + ".txt"
            for l in csv.reader(open(stat_filename, 'rU'), delimiter=' ') :
              #print l
              size_key = int(l[0])
              if size_key in speed :
                speed[size_key].append (map(float, l[1:]))
              else :
                speed[size_key] = []
                speed[size_key].append(map(float, l[1:]))
          
          speed_avgs = {size : map(np.average, zip(*speed[size]))  for size in speed}
          speed_stdevs = {size : map(np.std, zip(*speed[size]))  for size in speed}
          avg_std_zip = {size : [x for t in zip(speed_avgs[size], speed_stdevs[size]) for x in t] for size in speed }
          results = toLaTeX([ [size] + avg_std_zip[size] for size in sorted(avg_std_zip)])

          outfile = open("texfiles/speeds/" + "speeds_avgstdev_" + (dim_height + "_" if unfold else '') + ('_VM'+str(VMarkov) if VMarkov>0 else '') + ".tex", "wb")
          outfile.write(results)
          outfile.close()


# takes a list of lists (of numbers) and outputs latex-code for a table containing that numbers
def toLaTeX(numlist) :
  #res = "\\begin{tabular}{" + "c"*len(numlist[0]) + "}\n"
  #res +=  "\\\\\n"
  #res +=  "\\hline\n"
  res = ""
  for line in numlist :
    res += reduce(lambda x,y : x + " & " +'$' + str(round(y,2)) +'$' , line[1:], '$' + str(line[0]) +'$') + '\\\\\n'
  #res += "\\end{tabular}"
  return res


main()


#pcfg_speeds = [(float(x), float(y), float(z)) for [x,y,z] in csv.reader(open('speeds_42.txt', 'rU'), delimiter=' ') ]
#dim_speeds = [(float(x), float(y), float(z)) for [x,y,z] in csv.reader(open('speeds_D_42.txt', 'rU'), delimiter=' ') ]
#height_speeds = [(float(x), float(y), float(z)) for [x,y,z] in csv.reader(open('speeds_H_42.txt', 'rU'), delimiter=' ') ]
#parent_speeds = [(float(x), float(y), float(z)) for [x,y,z] in csv.reader(open('speeds_42_VM1.txt', 'rU'), delimiter=' ') ]
#dimpar_speeds = [(float(x), float(y), float(z)) for [x,y,z] in csv.reader(open('speeds_D_42_VM1.txt', 'rU'), delimiter=' ') ]
#heightpar_speeds = [(float(x), float(y), float(z)) for [x,y,z] in csv.reader(open('speeds_H_42_VM1.txt', 'rU'), delimiter=' ') ]
#
#pcfg_sizes = [(float(x), float(y)) for [x,y] in csv.reader(open('size_42.txt', 'rU'), delimiter=' ') ]
#dim_sizes = [(float(x), float(y)) for [x,y] in csv.reader(open('size_D_42.txt', 'rU'), delimiter=' ') ]
#height_sizes = [(float(x), float(y)) for [x,y] in csv.reader(open('size_H_42.txt', 'rU'), delimiter=' ') ]
#parent_sizes = [(float(x), float(y)) for [x,y] in csv.reader(open('size_42_VM1.txt', 'rU'), delimiter=' ') ]
#dimpar_sizes = [(float(x), float(y)) for [x,y] in csv.reader(open('size_D_42_VM1.txt', 'rU'), delimiter=' ') ]
#heightpar_sizes = [(float(x), float(y)) for [x,y] in csv.reader(open('size_H_42_VM1.txt', 'rU'), delimiter=' ') ]
#
#
#sample_sizes = list(set([x for (x,y,z) in pcfg_speeds]))
#sample_sizes.sort()
#
#short_round = partial(round, ndigits=2)
#
#avg_pcfg = map(short_round, [np.average([y for (x,y,z) in pcfg_speeds if x==i]) for i in sample_sizes])
#avg_dim = map(short_round, [np.average([y for (x,y,z) in dim_speeds if x==i]) for i in sample_sizes])
#avg_height = map(short_round, [np.average([y for (x,y,z) in height_speeds if x==i]) for i in sample_sizes])
#avg_parent = map(short_round, [np.average([y for (x,y,z) in parent_speeds if x==i]) for i in sample_sizes])
#avg_dimpar = map(short_round, [np.average([y for (x,y,z) in dimpar_speeds if x==i]) for i in sample_sizes])
#avg_heightpar = map(short_round, [np.average([y for (x,y,z) in heightpar_speeds if x==i]) for i in sample_sizes])
#
#avg_size_pcfg = map(short_round, [np.average([y for (x,y) in pcfg_sizes if x==i]) for i in sample_sizes])
#avg_size_dim = map(short_round, [np.average([y for (x,y) in dim_sizes if x==i]) for i in sample_sizes])
#avg_size_height = map(short_round, [np.average([y for (x,y) in height_sizes if x==i]) for i in sample_sizes])
#avg_size_parent = map(short_round, [np.average([y for (x,y) in parent_sizes if x==i]) for i in sample_sizes])
#avg_size_dimpar = map(short_round, [np.average([y for (x,y) in dimpar_sizes if x==i]) for i in sample_sizes])
#avg_size_heightpar = map(short_round, [np.average([y for (x,y) in heightpar_sizes if x==i]) for i in sample_sizes])


#factors = map(short_round, [x/y for (x,y) in zip(avg_dimpar, avg_dim)])

#size_factors = map(short_round, [x/y for (x,y) in zip(avg_size_height,avg_size_pcfg)])

#print "Avg pcfg speeds: " + str(avg_pcfg)
#print "Avg dim speeds: " + str(avg_dim)
#print "Avg height speeds: " + str(avg_height)
#print "Avg parent speeds: " + str(avg_parent)
#print "Avg dimpar speeds: " + str(avg_dimpar)
#print "Avg heightpar speeds: " + str(avg_heightpar)

#print "Speedup factors: " + str(factors)
#print "Size factors: " + str(size_factors)

#map(short_round, [x/y -1.0 for (x,y) in zip(avg_size_dim,avg_size_pcfg)])
#map(short_round, [x/y -1.0 for (x,y) in zip(avg_size_height,avg_size_pcfg)])
#map(short_round, [x/y -1.0 for (x,y) in zip(avg_size_parent,avg_size_pcfg)])
#map(short_round, [x/y -1.0 for (x,y) in zip(avg_size_dimpar,avg_size_pcfg)])
#map(short_round, [x/y -1.0 for (x,y) in zip(avg_size_heightpar,avg_size_pcfg)])
