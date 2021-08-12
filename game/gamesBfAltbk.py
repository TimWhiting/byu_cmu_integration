import csv
import random
import copy
from collections import Counter
#use this file

class gamesBf:
    def __init__(self, gameStr, me):
        gameSel = gameStr + "_" + str(me);
        self.name="BF"
        self.actions=["A_0","A_1","A_2"]
        self.states = self.readStatesFromFile("blocks2/Tables/A_S_" + gameSel + ".csv", 0)

        self.experts1= self.parseStateNames()#["s0_1", "s3_1", "s3_2"],["s1_0", "s2_1"]]


        self.priors = self.readFileNoHeader("blocks2/Tables/Prior_" + gameSel + ".csv")
        self.priorsOnly = self.readPriorsOnlyFromFile("blocks2/Tables/Prior_" + gameSel + ".csv")
        self.P_Z_S = self.readFile("blocks2/Tables/Z_S_" + gameSel + ".csv")
        self.P_S_Z_Z_S = self.readFile("blocks2/Tables/S_ZS_" + gameSel + ".csv")
        self.P_A_S = self.readFile("blocks2/Tables/A_S_" + gameSel + ".csv")
        self.P_A_S_noheader = self.readFileNoHeader("blocks2/Tables/A_S_" + gameSel + ".csv")

        self.P_S_A_A_S = self.readFile("blocks2/Tables/S_AS_" + gameSel + ".csv")
        self.playAction= None
        self.current_bayesian_values = self.getAggProbs(self.priorsOnly)  # aggregate the values for each state


    def calculateBelbarCurrentState(self, z_ssharp, z_partner, a_ssharp, a_partner, round, me, playerStr):  # part 2

        roundNumber = round
        probForAlpha = []
        for state in (self.states):
            sum = 0
            for newState in self.states:
                prob = 0
                thisPrior = 0
                # find index of state
                for i in range(len(self.P_S_Z_Z_S[0])):
                    if (self.P_S_Z_Z_S[0][i] == state):
                        thisIndex = i
                        break

                # prob is the probability we'll transfer to newState given state, z_i, z_noti
                for trans in self.P_S_Z_Z_S:
                    if (trans[0] == newState and trans[1 + me] == z_ssharp and trans[1 + (1 - me)] ==
                            z_partner):
                        prob = float(trans[thisIndex])
                        break

                for prior in self.priors:
                    if (prior[0] == newState):
                        thisPrior = prior[1]
                        break
                sum += prob * thisPrior
            probForAlpha.append(sum)  # the probability I'm going to state state
        # updating the priors here
        i = 0
        total = 0
        for prior in self.priors:
            prior[1] = probForAlpha[i]
            total += prior[1]
            self.priorsOnly[i] = prior[1]
            i += 1

        normalized = self.priorsOnly;
        mag = 0.0;
        for i in range(0, len(normalized)):
            normalized[i] = normalized[i] / total;
            mag = mag + normalized[i]

        with open("blocks2/output/probs_" + playerStr + "_blocks2_" + str(me) + ".csv",
                  "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([roundNumber] + normalized)  # self.priorsOnly)
        output.close()

        self.calculateBelCurrentState(a_ssharp, a_partner, z_ssharp, roundNumber, me,
                                          playerStr)


    def calculateBelCurrentState(self, action_ssharp, action_partner, z_ssharp, roundNumber, me,
                                 playerStr):  # action of s# only, part2 ko part2
        belHat = []
        index = 0
        i = 0
        sum = 0
        act = "A_" + str(action_ssharp)
        # find index of the given action
        for i in range(len(self.P_A_S[0])):
            if (self.P_A_S[0][i] == act):
                index = i
                break
        ########print("Index of given action S#: ", act, "is: ", index, '\n')
        i = 0
        for state in self.states:
            for pas in self.P_A_S:
                if (pas[0] == state):  # pzs[index] == obser1

                    sum += float(pas[index]) * float(self.priorsOnly[i])  #
                    belHat.append(float(pas[index]) * float(self.priorsOnly[i]))
            i = i + 1
        ########print("belhat at time t=0 for action:",act,"is:", belHat)
        alpha = 1 / sum
        ########print("Alpha is", alpha)
        i = 0
        total = 0
        for prior in self.priors:
            prior[1] = belHat[i] * alpha
            total += prior[1]
            self.priorsOnly[i] = prior[1]
            i += 1
        # print('\n',"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PART 2 end results  bel hat%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",'\n')
        # print("\t", "Results:3 part 2 step 2 New Priors=", self.priors)#prints the final bel hat for part 2
        # print("\t", "New Priors=", self.priors,'\n')#prints the final bel hat for part 2

        normalized = self.priorsOnly;
        # print(total)
        mag = 0.0;
        for i in range(0, len(normalized)):
            normalized[i] = normalized[i] / total;
            mag = mag + normalized[i]
        # print(mag)
        #print("prior", self.priorsOnly)
        #print("norm prior",normalized)
        with open("blocks2/output/probs_" + playerStr + "_blocks2_" + str(me) + ".csv",
                  "a") as output:  # writes the third line, happens for all other rounds
            writer = csv.writer(output, lineterminator='\n')
            # writer.writerow(['Round','states'])
            writer.writerow([roundNumber] + self.priorsOnly)
        output.close()
        #august use this prior for prob distribution
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PART 2 end results  bel hat%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",'\n')
        ########print("****************************************")
        maxindex = self.priorsOnly.index(max(self.priorsOnly))
        # print("Part2 end result Max prior is: ", max(self.priorsOnly), "in state:", self.priors[maxindex][0])

        if (z_ssharp != "null"):
            #august, call def actions to find out what action to take in next step- action prediction
            self.playAction= self.calcPA(playerStr,me)#this is the action taken by player for next round
            posteriorscopy= copy.deepcopy(self.priorsOnly)
            self.current_bayesian_values= self.getAggProbs(posteriorscopy)#aggregate the values for each state
            self.calculateBelbarNextState(action_ssharp, action_partner, z_ssharp, roundNumber, me, playerStr)

    def getAggProbs(self,postprobs):
        keys=self.states
        vals=postprobs
        dictStates = dict(zip(keys, vals))
        aggprobs=[]
        for expertgroup in self.experts1:
            sum = 0
            for exp in expertgroup:
                if exp in dictStates.keys():
                    sum+= float(dictStates.get(exp))
            aggprobs.append(sum)

        return aggprobs

    def parseStateNames(self):
        experts=[]
        explist = []
        explist1 = []
        #print("length is: ", len(self.states))
        for val in self.states:
            #print("here:", val)
            expert, state = val.split("_")
            explist.append(expert)
        exp = Counter(explist).keys()
        expKeyList=list(exp)
        #print("experts in list", expKeyList)
        count = Counter(explist).values()
        for i in range(len(expKeyList)):
            for j in range(len(self.states)):
                #print("states j:", self.states[j])
                #print("exp:", expKeyList[i])
                s1 = self.states[j].split("_")
                #print("s1", s1[0])
                if s1[0] == expKeyList[i]:
                    explist1.append(self.states[j])
            experts.append(explist1)
            explist1 = []
        #print(self.experts)
        return experts

    def calculateBelbarNextState(self, act_ssharp, act_partner, z_ssharp, roundNumber, me, playerStr):  # part 3
        # print("Act S#:",act_ssharp)
        # for cur in range(len(act_ssharp)):
        probForAlpha = []

        for state in (self.states):
            sum = 0

            for i in range(len(self.P_S_A_A_S[0])):
                if (self.P_S_A_A_S[0][i] == state):
                    thisIndex = i
                    break

            i = 0
            for trans in self.P_S_A_A_S:
                if (trans[1 + me] == act_ssharp and trans[1 + (1 - me)] == act_partner):
                    sum += self.priorsOnly[i] * float(trans[thisIndex])
                    i = i + 1

            probForAlpha.append(sum)

        # updating the priors here
        i = 0
        total = 0
        for prior in self.priors:
            prior[1] = probForAlpha[i]
            total += prior[1]
            self.priorsOnly[i] = prior[1]
            i += 1
        ########print("\t", "Results:4 intermediate, part 3 step1 New Priors=", self.priors)
        ########print("****************************************")

        normalized = self.priorsOnly;
        #print(total)
        mag = 0.0;
        for i in range(0, len(normalized)):
            normalized[i] = normalized[i] / total;
            mag = mag + normalized[i]
        # print(mag)

        with open("blocks2/output/probs_" + playerStr + "_blocks2_" + str(me) + ".csv",
                  "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            r = str(int(roundNumber) + 1)
            writer.writerow([r] + normalized)  # self.priorsOnly)
        output.close()
        # print("For sanity check Total: ", total)
        # print("Priors only", self.priorsOnly)
        self.calculateBelNextState(z_ssharp, roundNumber, me, playerStr)

    def calculateBelNextState(self, z_ssharp, roundNumber, me, playerStr):
        belHat = []
        index = -1
        i = 0
        sum = 0

        #print("\n", roundNumber, z_ssharp)

        # find index of the given action
        for i in range(len(self.P_Z_S[0])):
            if (self.P_Z_S[0][i] == z_ssharp):
                #print(self.P_Z_S[0][i], '\n') #commented
                index = i
                break

        # print("Index is: ", index, "\n")
        i = 0
        for state in self.states:
            for pzs in self.P_Z_S:
                if (pzs[0] == state):  # pzs[index] == obser1
                    sum += float(pzs[index]) * float(self.priorsOnly[i])
                    belHat.append(float(pzs[index]) * float(self.priorsOnly[i]))
                    # print(state,pzs[index],self.priorsOnly[i],belHat[i])
            i = i + 1
        ########print("belhat at time t=1 for S# proposal:", znoti, "is:", belHat)
        alpha = 1 / sum
        ########print("Alpha is", alpha)
        i = 0
        total = 0
        for prior in self.priors:
            prior[1] = belHat[i] * alpha
            total += prior[1]
            self.priorsOnly[i] = prior[1]
            i += 1
        # print('\n', "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PART 3 end results  bel hat%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",  '\n')
        # print("\t", "Results:5 part 3 step 2 New Priors=", self.priors)#prints end result for part 3, check here if priors or priors only

        with open("blocks2/output/probs_" + playerStr + "_blocks2_" + str(me) + ".csv",
                  "a") as output:  # the fourth line, for all other rounds
            writer = csv.writer(output, lineterminator='\n')
            r = str(int(roundNumber) + 1)
            writer.writerow([r] + self.priorsOnly)
        output.close()
        # print('\n',  "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PART 3 end results  bel hat%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",'\n')
        ########print("****************************************")
        maxindex = self.priorsOnly.index(max(self.priorsOnly))
        # print("Part3 end result Max prior is: ", max(self.priorsOnly),"in state:",self.priors[maxindex][0])

    def calculatePriorBelief(self, obser1, playerStr,me):  # part1
        belHat = []
        index = 0
        i = 0
        sum = 0
        with open("blocks2/output/probs_" + playerStr + "_blocks2_" + str(me) + ".csv",
                  "a") as output:  # writes the 1st line in the fine, the Round and state names, only once
            writer = csv.writer(output)
            writer.writerow(['Round'] + self.states)
        output.close()

#dec 9 najma why is this here and at bottom as well?, priors written twice i think, this is not required
        #print("1: this prior", self.priorsOnly)
        with open("blocks2/output/probs_" + playerStr + "_blocks2_" + str(me) + ".csv",
                  "a") as output:
            writer = csv.writer(output)
            writer.writerow(['0']+ self.priorsOnly)
        output.close()


        # find index of the given action
        for i in range(len(self.P_Z_S[0])):
            if (self.P_Z_S[0][i] == obser1):
                index = i  #####note: what if the proposed solution is not among the one we have in list/ table
                break

        # print(index);

        ########print("Index of given observation", obser1,"is: ",index,'\n')
        i = 0
        for state in self.states:
            for pzs in self.P_Z_S:
                if (pzs[0] == state):  # pzs[index] == obser1
                    sum += float(pzs[index]) * float(self.priorsOnly[i])
                    belHat.append(float(pzs[index]) * float(self.priorsOnly[i]))
            i = i + 1
        ########print("Belhat at time t=0", belHat)
        alpha = 1 / sum
        ########print("Alpha is",alpha)
        i = 0
        total = 0
        for prior in self.priors:
            prior[1] = belHat[i] * alpha
            total += prior[1]
            self.priorsOnly[i] = prior[1]
            i += 1
        #maxindex = self.priorsOnly.index(max(self.priorsOnly))

        normalized = self.priorsOnly;
        #print(total)
        mag = 0.0;
        for i in range(0,len(normalized)):
            normalized[i] = normalized[i] / total;
            mag = mag + normalized[i]
        #print(mag)

        with open("blocks2/output/probs_" + playerStr + "_blocks2_" + str(me) + ".csv",
                  "a") as output:
            writer = csv.writer(output)
            writer.writerow(['0']+ normalized)#self.priorsOnly)
        output.close()
        ########print("****************************************")
        # print("For sanity check Total: ", total)
        # print("Priors only", self.priorsOnly)


    def calcPA(self,playerStr,me ):
        total = []
        text = []
        maxm = 0
        #for loop in range(len(self.P_S)):
        #print("Round: ", self.P_S[loop][0])
        #print("Loop:", loop)
        #text.append(self.P_S[loop][0])
        #text.append(loop)
        for action in self.actions:
            sum = 0
            a = self.Index(action)
            for row, state in enumerate(self.states):
                #print(len(self.states))
                #print("row",row)
                #print("state", state)
                x = self.Index(state)
                #print("x",x)
                #print("P_S value:", float(self.priorsOnly[x]))
                #print("P_A_S: ", float(self.P_A_S_noheader[row][a]))#plus one to ignore header
                #print("\n")

                sum = sum + float(self.priorsOnly[x]) * float(self.P_A_S_noheader[row][a])
            total.append(sum)
        #print("Total: ", total)
        text.append(total[0])
        text.append(total[1])
        text.append(total[2])
        maxm= total.index(max(total))
        text.append(maxm)
        with open(
                "blocks2/outputAction/actions_" + playerStr + "_blocks2_" + str(me) + ".csv",
                "a") as output:
            writer = csv.writer(output)
            writer.writerow(text)
        output.close()
        text = []
        total = []
        return maxm

        # print("Prob of a- 0 and a-1 sequentially:",total)
        # print("len:", len(total))

    def readStatesFromFile(self, filename, i):
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            data = [r[i] for r in reader]
        return data

    def readPriorsOnlyFromFile(self, filename):
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            data = [r[1] for r in reader]
        return data

    def readFile(self, filename):
        with open(filename) as f:
            reader = csv.reader(f)
            # next(reader)  # skip header
            data = [r for r in reader]
        return data
        # print(data)

    def readFileNoHeader(self, filename):
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            data = [r for r in reader]
        return data

    def printStates(self):
        print("States:", '\n')
        print('\t', self.states)
        print("***********************************")
        print("Priors:", '\n')
        # print('\t',self.priors)
        for prior in self.priors:
            print('\t', prior, '\n')
        print("***********************************")
        print("Prob of seeing a proposal given a state [P(Z|S)]:", '\n')
        for pzs in self.P_Z_S:
            print('\t', pzs, '\n')
        print("***********************************")
        print("Prob of moving to a state given Z(s#), Z(human), state [P(S|Z,Z,S)]:", '\n')
        for ps_zzs in self.P_S_Z_Z_S:
            print('\t', ps_zzs, '\n')
        print("***********************************")

        print("Prob of taking an action by S#  given a state(s#) [P(A|S)]:", '\n')
        for pas in self.P_A_S:
            print('\t', pas, '\n')
        print("***********************************")

    def Index(self, value):
        # checkAction= False
        if (value[0] == 'A'):
            x = self.actions
            # checkAction=True
            for index, s in enumerate(x):
                if (s == value):
                    return index +1
        else:
            x = self.states
            for index, s in enumerate(x):
                if (s == value):
                    #return index + 1 #august najma removed +1 because we dont have rounds in first column
                    return index


def readFilenames(filename):
    file = open(filename, 'r')
    txt = file.read().splitlines()  # without /n
    print("here text is:", txt)
    return txt


def playerNames(name):
    s1 = name.find('_')
    print("s1", s1)
    s2 = name.find('_', s1 + 1)
    print("s2", s2)
    s3 = name.find('_', s2 + 1)
    print(s3)
    length = s3 - s1
    s = name[s1 + 1: s1 + length]
    print(s)
    return s

'''
### Start of the program
# playerStr = sys.argv[1]
# gameStr = sys.argv[2]
gameStr = "blocks2"
# me = int(sys.argv[3])
me = 0#me is the player being modeled, which is the human player
bf = gamesBf(gameStr, me)


playerStr=""

observations_z0="None"
observations_z1="None"
action0=str(random.randint(0,2))# the code has to model this action
action1=str(random.randint(0,2))#the code plays this action
print("Player: "+ str(me)+" played: "+ action0)
print("Player: "+ str(me)+" played: "+ action1)
bf.calculatePriorBelief("None", playerStr)  # initialProposal is the 1st S# proposal
rounds=20
for i in range(0,rounds):
    print("round",i)
    if (me == 0):
        bf.calculateBelbarCurrentState(observations_z0, observations_z1, action0, action1, i, me, playerStr)
    #else:
        #bf.calculateBelbarCurrentState(observations_z1, observations_z0, action1, action0, i, me, playerStr)
    action1=str(bf.playAction)#call funtion doing map over actions
    print("modeled action", action1)
    #action0=str(random.randint(0,2))
    action0= str(input("enter choice"))
    print("Player: " + str(me) + " played: " + action0)
    print("Player: " + str("1") + " played: " + action1)
'''