# collate all results and plot
from statistics import mean
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


occupations = ["biologist","ceo","cook","engineer","nurse","police_officer","primary_school_teacher","programmer","software_developer","truck_driver"]

def main():
    overall_relevance = {}
    overall_fairness = {}
    for occupation in occupations:
        with open("./profession/"+occupation+"/results.txt") as f:
            for line in f:
                line = line.strip()
                d = line.split(",")
                system_tag =  d[0]
                relevance = float(d[1])
                fairness = float(d[2])
                
                m = overall_relevance.get(system_tag,[])
                m.append(relevance)
                overall_relevance[system_tag] = m
                
                n = overall_relevance.get(system_tag,[])
                n.append(fairness)
                overall_relevance[system_tag] = n
    
    systems = overall_relevance.keys()
    systems.sort()
    with open("./profession/overall_results.txt","w") as f:
        for system in systems:
            f.write(system+","+str(mean(overall_relevance[system]))+","+str(mean(overall_fairness[system]))+"\n")
    
    # now plot
    colormap = plt.cm.gist_ncar 
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(systems))]
    for i, run in enumerate(systems):
        plt.scatter(x=[mean(overall_fairness[run])], 
            y=[mean(overall_relevance[run])], c=[colors[i]],
            label=run, s=50)

    plt.legend(loc='center', fontsize='xx-small',bbox_to_anchor=(0.5, 1.05),ncol=len(systems)-3)
    plt.xlabel("Unfairness (NDKL)")
    plt.ylabel("Relevance")
    plt.savefig('./profession/overall-plot.pdf')
    
if __name__=="__main__":
	main()
                
