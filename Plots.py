import csv
import matplotlib.pyplot as plt

def Bound(a, T):
	A = []
	A.append(1)
	for i in range(1, T+1):
		Ai = ((a*(2-a)*A[-1]) - (a*(1-a)*A[-1]*A[-1]))/(1-a+(a*A[-1]))
		A.append(Ai)

	# print(A)

	sum = 0
	for i in range(0, T+1):
		sum += A[i]

	S = (1-a)/(1- 3*a + a*a)
	ExpLead = sum + (S/1000000000)
	FracLead = 1 - (1/ExpLead)

	return FracLead

T = 3000
power = []
SimBounds = []
OldBounds = []
RecentBounds = []
SOBounds = []
OneLook = []
for d in range(0, 291):
	a = d/1000
	power.append(a)
	SimBounds.append(Bound(a, T)-a)
	b = a*(2-a)/(1-a)
	br = a*(1-a)/(1-2*a)
	bs = (a - 2*a*a + a*a*a - a*a*a*a)/(1 - 3*a + 3*a*a - 3*a*a*a)
	OldBounds.append(b-a)
	RecentBounds.append(br-a)
	SOBounds.append(bs-a)
	sm = sum((a ** i) * (1 + i * a / (1 + (i-1) * a)) for i in range(1, 1000))
	OneLook.append((1-a)/(1+a) * sm - a)
	print("a: ", power[-1], "New Bound: ", SimBounds[-1], "Recent Bound: ", RecentBounds[-1], "Old Bound: ", OldBounds[-1])

f = open('bounds', 'w')
writer = csv.writer(f)

writer.writerow(power)
writer.writerow(SimBounds)
writer.writerow(RecentBounds)
writer.writerow(OldBounds)

alphas = [i/100 for i in range(1, 30)]

beta0_lower_raw = [
	0.009900875631,
	0.01993197266,
	0.02992691517,
	0.03994765568,
	0.04994391104,
	0.0599691255,
	0.06997851652,
	0.07998814071,
	0.09006135737,
	0.1001128302,
	0.1101472255,
	0.1202451767,
	0.1303593337,
	0.1404470886,
	0.1505488429,
	0.1607251036,
	0.1708458267,
	0.1810278074,
	0.1912181223,
	0.2014787002,
	0.2116823015,
	0.2219957528,
	0.232278063,
	0.2425986126,
	0.252952893,
	0.2633550401,
	0.2737645875,
	0.2841917746,
	0.2946707483
]

beta0_upper_raw = [
	0.01008419501,
	0.02005181442,
	0.0301002968,
	0.04009583248,
	0.05013142909,
	0.06011889049,
	0.07014662485,
	0.08021367417,
	0.09022086477,
	0.1002655091,
	0.110254117,
	0.1203740958,
	0.1305259905,
	0.140553322,
	0.1507065732,
	0.160877303,
	0.1709560486,
	0.1811940648,
	0.1913731628,
	0.2015711942,
	0.2118654795,
	0.2221266046,
	0.2323953144,
	0.2427265613,
	0.2530064592,
	0.2634539642,
	0.2738794858,
	0.2843231806,
	0.2948422916
]

beta5_lower_raw = [
	0.009928943904,
	0.01996528364,
	0.0298859244,
	0.03986732103,
	0.04991782756,
	0.05993048549,
	0.07005075207,
	0.08004354747,
	0.0900235043,
	0.1000726539,
	0.1102644814,
	0.1203883013,
	0.130520927,
	0.1406240379,
	0.1508738176,
	0.1610062242,
	0.171243448,
	0.1816945524,
	0.1920109238,
	0.202405492,
	0.2127554974,
	0.2232149249,
	0.2337517929,
	0.2442816033,
	0.2549864223,
	0.2656148033,
	0.2762225401,
	0.2870195995,
	0.297779746
]

beta5_upper_raw = [
	0.01017981546,
	0.02006858395,
	0.03019819211,
	0.04020497246,
	0.05020898409,
	0.06023128619,
	0.07029196718,
	0.08031279048,
	0.09041135283,
	0.1003884035,
	0.1106171874,
	0.1207121927,
	0.1308629341,
	0.1410505406,
	0.1512553034,
	0.1614008845,
	0.1716081481,
	0.18204855,
	0.1924053253,
	0.2027587233,
	0.2131465013,
	0.2235441349,
	0.2341557134,
	0.2447302622,
	0.255424438,
	0.2660739323,
	0.2767346216,
	0.2874965401,
	0.2982572688
]

beta1_lower_raw = [
	0.009891312618,
	0.01991856045,
	0.02994446821956568,
	0.04001489200014899,
	0.050013037990321696,
	0.06017661088,
	0.07028672363,
	0.08045381235,
	0.09062634403,
	0.100938986,
	0.111305989,
	0.121655785,
	0.1321031638,
	0.1425848267,
	0.153190517,
	0.163783517,
	0.1745762547,
	0.185381878,
	0.1962512873,
	0.2072754154,
	0.2182167176,
	0.2294332676,
	0.2406482242,
	0.2519658249,
	0.2634021763,
	0.2748911676,
	0.2864760578,
	0.2982718486,
	0.3100631108
]

beta1_upper_raw = [
	0.01007897713,
	0.02008827107,
	0.030175271564887352,
	0.04016764529685006,
	0.05026715088168982,
	0.06032864042,
	0.07045785787,
	0.08062070801,
	0.0908144293,
	0.1011224448,
	0.1114291196,
	0.1217796834,
	0.1322155453,
	0.1427446558,
	0.1533048884,
	0.1639770307,
	0.1746970561,
	0.1855053986,
	0.196383743,
	0.2073420575,
	0.2184461967,
	0.2295549904,
	0.2407875803,
	0.2521093058,
	0.263584692,
	0.2749942352,
	0.2866813375,
	0.2982794178,
	0.3101351471
]

betas = [i/25 for i in range(0, 26)]

alpha25_lower_raw = [
	0.2529234149,
	0.2522675023,
	0.2523304179,
	0.2524563372,
	0.2528608255,
	0.2528732275,
	0.2532383423,
	0.2532795805,
	0.2532907035,
	0.2535548058,
	0.2539823102,
	0.25381407,
	0.2542316791,
	0.254518937,
	0.2549020085,
	0.2551832394,
	0.2555266489,
	0.2561335121,
	0.2563745533,
	0.2570187423,
	0.2577139142,
	0.2582787147,
	0.2591647299,
	0.2602998497,
	0.2610733687,
	0.2631923059
]

alpha25_upper_raw = [
	0.2532207012,
	0.2537244252,
	0.2538230371,
	0.2540811953,
	0.2539825645,
	0.2541072692,
	0.2544743816,
	0.2546179552,
	0.2547952462,
	0.2549439587,
	0.2551094458,
	0.255295178,
	0.2558709972,
	0.2560326259,
	0.2564028961,
	0.2564643485,
	0.2571520172,
	0.2575532467,
	0.2581886361,
	0.2585160112,
	0.2593385273,
	0.2600759752,
	0.2610541723,
	0.2617926977,
	0.2629569557,
	0.2635052933
]

beta0_lower = [beta0_lower_raw[i] - alphas[i] for i in range(29)]
beta0_upper = [beta0_upper_raw[i] - alphas[i] for i in range(29)]
beta5_lower = [beta5_lower_raw[i] - alphas[i] for i in range(29)]
beta5_upper = [beta5_upper_raw[i] - alphas[i] for i in range(29)]
beta1_lower = [beta1_lower_raw[i] - alphas[i] for i in range(29)]
beta1_upper = [beta1_upper_raw[i] - alphas[i] for i in range(29)]
alpha25_lower = [alpha25_lower_raw[i] - 0.25 for i in range(26)]
alpha25_upper = [alpha25_upper_raw[i] - 0.25 for i in range(26)]

# plot 1
plt.plot(power, SimBounds, label = "Tight omniscient bound")
plt.plot(power, OldBounds, label = "Bound from [FHWY22]")
plt.plot(alphas, beta1_upper, label = "Computational upper bound")
plt.plot(power, OneLook, label = "1-Lookahead Rewards")
plt.xlabel("Stake")
plt.ylabel("Reward - Stake")

# plot 2
#plt.plot(alphas, beta0_lower, label = "Beta=0, lower bound")
#plt.plot(alphas, beta0_upper, label = "Beta=0, upper bound")
#plt.plot(alphas, beta5_lower, label = "Beta=0.5, lower bound")
#plt.plot(alphas, beta5_upper, label = "Beta=0.5, upper bound")
#plt.plot(alphas, beta1_lower, label = "Beta=1, lower bound")
#plt.plot(alphas, beta1_upper, label = "Beta=1, upper bound")
#plt.xlabel("Stake")
#plt.ylabel("Reward - Stake")

# plot 3
#plt.plot(betas, alpha25_lower, label = "Lower bound")
#plt.plot(betas, alpha25_upper, label = "Upper bound")
#plt.xlabel("Beta")
#plt.ylabel("Reward - Stake")

plt.legend()
plt.show()