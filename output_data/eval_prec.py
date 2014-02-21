import numpy as np
import csv
from functools import partial


bank = "TueBaDZ"

eval_methods = ["evalb", "LA", "CBEval"]
seeds = [1271,49647,53316,25907,13023,898,30964,51253,54469,45047]


def main() :
  header = []
  learning = {}
  for method in eval_methods :
    for dim_height in ["N", "D", "H"] :
      for VMarkov in [0,1] :
        if method == "evalb" :
          header.append(dim_height + ('-VM'+str(VMarkov) if VMarkov>0 else ''))
        prec = {}
        for seed in seeds:
          stat_filename = bank + "_" + method + "_stats_" + (dim_height + "_" if dim_height != "N" else '') + str(seed) + ('_VM'+str(VMarkov) if VMarkov>0 else '') + ".txt"
          for l in csv.reader(open(stat_filename, 'rU'), delimiter=' ') :
            #print l
            size_key = int(l[0])
            if size_key in prec :
              prec[size_key].append (map(float, l[1:]))
            else :
              prec[size_key] = []
              prec[size_key].append(map(float, l[1:]))

        prec_avgs = {size : map(np.average, zip(*prec[size]))  for size in prec}
        prec_stdevs = {size : map(np.std, zip(*prec[size]))  for size in prec}
        avg_std_zip = {size : [x for t in zip(prec_avgs[size], prec_stdevs[size]) for x in t] for size in prec }
        
        if method == "evalb" :
          for size in prec_avgs :
            if size in learning :
              learning[size] += [ prec_avgs[size][2],prec_stdevs[size][2] ]
            else :
              learning[size] = [ prec_avgs[size][2],prec_stdevs[size][2] ]

        results = toLaTeX([ [size] + avg_std_zip[size] for size in sorted(avg_std_zip)])
        outfile = open("texfiles/" + bank + "_" + method + "_avgstdev_stats_" + (dim_height + "_" if dim_height != "N" else '') + ('_VM'+str(VMarkov) if VMarkov>0 else '') + ".tex", "wb")
        outfile.write(results)
        outfile.close()

  learning_file = open("texfiles/learning-curves.txt","wb")
  learning_file.write("size ")
  for x in header :
    learning_file.write(x + "-mean " + x + "-stdev ")
  learning_file.write('\n')
  res = ""
  for size in sorted(learning) :
    res += reduce(lambda x,y : x + " " + str(round(y,2)), learning[size], str(size)) + '\n'
  learning_file.write(res)
  learning_file.close()

  r = get_prec_by_method()
  prec_file = open("texfiles/prec.tex","wb")
  prec_file.write(dict2Latex(r))
  prec_file.close()

  s = get_speedsize()
  speed_file = open("texfiles/speeds.tex","wb")
  speed_file.write(dict2Latex(s))
  speed_file.close()

  return

def get_speedsize(std=False) :
  stats = {} #for every config this holds a list of averaged metrics ( F_1,exact(evalb) , LA_corp,LA_sent, CB,Zero_CB)
  for dim_height in ["Plain", "D", "H"] :
    for VMarkov in [0,1] :
      config = dim_height + ('-VM'+str(VMarkov) if VMarkov>0 else '')
      data = [] #list of lists (of metrics)
      for seed in seeds:
        line = []
        speed_filename = "speeds_" + (dim_height + "_" if dim_height != "Plain" else '') + str(seed) + ('_VM'+str(VMarkov) if VMarkov>0 else '') + ".txt"
        size_filename = "size_" + (dim_height + "_" if dim_height != "Plain" else '') + str(seed) + ('_VM'+str(VMarkov) if VMarkov>0 else '') + ".txt"
        for l in csv.reader(open(size_filename, 'rU'), delimiter=' ') :
          if int(l[0]) == 70000 : # we are only interested in the stats for the largest sample size
            #print l
            line += [float(l[1])]
        for l in csv.reader(open(speed_filename, 'rU'), delimiter=' ') :
          if int(l[0]) == 70000 : # we are only interested in the stats for the largest sample size
            #print l
            line += map(float,[l[2]]) #1 = wds/sec, #2 = sents/sec 
        data.append(line)
      #print data
      stats_avgs = map(np.average, zip(*data)) #column-wise average and std-deviation
      stats_stdevs = map(np.std, zip(*data))
      if std :
        stats[config] = [x for t in zip(stats_avgs, stats_stdevs) for x in t]
      else :
        stats[config] = stats_avgs
  return stats

# get statistics for the 70k sample (averaged over all trials)
def get_prec_by_method(std=False) :
  prec = {} #for every config this holds a list of averaged metrics ( F_1,exact(evalb) , LA_corp,LA_sent, CB,Zero_CB)
  for dim_height in ["Plain", "D", "H"] :
    for VMarkov in [0,1] :
      config = dim_height + ('-VM'+str(VMarkov) if VMarkov>0 else '')
      precs = [] #list of lists (of metrics)
      for seed in seeds:
        line = []
        for method in eval_methods:
          stat_filename = bank + "_" + method + "_stats_" + (dim_height + "_" if dim_height != "Plain" else '') + str(seed) + ('_VM'+str(VMarkov) if VMarkov>0 else '') + ".txt"
          for l in csv.reader(open(stat_filename, 'rU'), delimiter=' ') :
            if int(l[0]) == 70000 : # we are only interested in the stats for the largest sample size
              #print l
              if method == "evalb" :
                line += map(float,[l[3], l[4]]) #3 = F_1, 4=exact
              elif method == "LA":
                line += map(float,[l[1], l[3]]) #1 = LA (sentence), 3 = LA (corpus)
              else : #CBEval
                line += map(lambda x : x / 100.0, map(float,[l[1], l[2]])) #1 = avg # of CB, 2 = Zero_CB (strangely both are given in % !)
        precs.append(line)
      #print precs
      prec_avgs = map(np.average, zip(*precs)) #column-wise average and std-deviation
      if std :
        prec_stdevs = map(np.std, zip(*precs))
        prec[config] = [x for t in zip(prec_avgs, prec_stdevs) for x in t]
      else :
        prec[config] = prec_avgs
  return prec

def dict2Latex(d,acc=2) :
  res = ""
  for key in d :
    res += str(key) + " & " + reduce(lambda x,y : x + " & " +'$' + str(round(y,acc)) +'$' , d[key][1:], '$' + str(round(d[key][0],2)) +'$') + '\\\\\n'
  return res

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


