import random, time, sys
from pprint import pprint
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter

ships = {
    'D' : 2,
    'C' : 3,
    'S' : 3,
    'B' : 4,
    'A' : 5
    }

def check_crash (bg, s, r, c, hori):
    if hori == False:
        for i in range (s):
            if r+i < 0 or r+i > 9:
                return True
            elif bg[r+i][c] != '.' and bg[r+i][c] != '*':
                # In hunt mode '*' on yourBoard is allowed placement like a ' '
                # Currently '*' (partial hit) not possible in radar mode
                # If changed, disallow '*' if radar mode
                return True
        return False
    else:
        for i in range (s):
            if c+i < 0 or c+i > 9:
                return True
            elif bg[r][c+i] != '.' and bg[r][c+i] != '*':
                return True
        return False

def count_shots(bg):
    shots = 0
    for i in range (10):
        for j in range (10):
            if bg[i][j] != '.':
                shots += 1
    print(shots, 'number of shots')

def currentMilliTime():
    return round(time.time() * 1000)

class ai:
    def __init__(self):
        # My real board
        # Ship character to indicate ship placement by user
        self.myBoard = []

        # My understanding of your board
        # ' ' for not checked
        # '_' for misses
        # Ship character for sunk ships
        # '*' for partial hits ignored by checkCrash to fit ships for probCalc in hunt mode
        self.yourBoard = []
        for i in range(10):
            self.myBoard.append(['.']*10)
            self.yourBoard.append(['.']*10)

        # Which of your ships have I hit
        self.myHits = {
            'D' : 0,
            'C' : 0,
            'S' : 0,
            'B' : 0,
            'A' : 0
            }

        """
        self.coords = {
            'D' : [],
            'C' : [],
            'S' : [],
            'B' : [],
            'A' : []
            }

        # How many of your ships have I sunk
        self.yourSunk = 0

        # Which of your ships have I hit but not sunk
        self.activeYourHits = []
        """

        self.unresolvedHits = []

        self.sunkShips = {}

    def placeShips(self):
        ship_list = list(ships.keys())
        for s in ship_list:
            while True:
                hori = random.choice([True, False])
                if hori == True: # Hori
                    r = random.randint(0, 9)
                    c = random.randint(0, (10 - ships[s]))

                else: # Vert
                    r = random.randint(0, (10 - ships[s]))
                    c = random.randint(0, 9)

                if check_crash(self.myBoard, ships[s], r, c, hori) == False:
                    break
                else:
                    # print (s, "clash detected! Retrying.")
                    r = r

            for i in range (ships[s]):
                if hori == False:
                    self.myBoard[r+i][c] = s
                else:
                    self.myBoard[r][c+i] = s

    def checkShipPoss(self, s, r, c): # r and c are based on 'sunk' position
        posi = []
        offset = ships[s]
        start = 0
        end = 0

        # Vertical check
        for i in range(r, r-offset, -1):
            if((i < 0 or i > 9) or self.yourBoard[i][c] != '*'):
                break
            start = i

        for i in range(r, r+offset):
            if((i < 0 or i > 9) or self.yourBoard[i][c] != '*'):
                break
            end = i

        if(end - start >= offset-1):
            for i in range(start, end-(offset-2)):
                posi.append([offset, i, c, False])

        # Horizontal check
        start = 0
        end = 0

        for i in range(c, c-offset, -1):
            if((i < 0 or i > 9) or self.yourBoard[r][i] != '*'):
                break
            start = i

        for i in range(c, c+offset):
            if((i < 0 or i > 9) or self.yourBoard[r][i] != '*'):
                break
            end = i

        if(end - start >= offset-1):
            for i in range(start, end-(offset-2)):
                posi.append([offset, r, i, True])

        return posi

    def probCalculate(self):
        maxi = 0
        bg = []
        for i in range(10):
            bg.append([0]*10)

        ship_list = list(ships.keys())
        # print(ship_list)
        if self.unresolvedHits:
            for unresHit in self.unresolvedHits:
                if unresHit in self.sunkShips.values():
                    continue
                for s in ship_list:
                    if s in self.sunkShips:
                        continue
                    for hori in [True, False]:
                        if hori == True: # Hori
                            for c in range (unresHit[1]-ships[s]+1, unresHit[1]+1):
                                #print(active_hit, s, hori, c)
                                if check_crash(self.yourBoard, ships[s], unresHit[0], c, hori) == False:
                                    for k in range (ships[s]):
                                        if self.yourBoard[unresHit[0]][c+k] != '*':
                                            bg[unresHit[0]][c+k] += 1
                                            if bg[unresHit[0]][c+k] > maxi:
                                                maxi = bg[unresHit[0]][c+k]
                                                max_i = c+k
                                                max_j = unresHit[0]
                        else:
                            for r in range (unresHit[0]-ships[s]+1, unresHit[0]+1):
                                #print (active_hit, s, hori, r)
                                if check_crash(self.yourBoard, ships[s], r, unresHit[1], hori) == False:
                                    for k in range (ships[s]):
                                        if self.yourBoard[r+k][unresHit[1]] != '*':
                                            bg[r+k][unresHit[1]] += 1
                                            if bg[r+k][unresHit[1]] > maxi:
                                                maxi = bg[r+k][unresHit[1]]
                                                max_i = unresHit[1]
                                                max_j = r+k
        else:
            #print('Radar scanning in search mode.')
            for s in ship_list:
                if s in self.sunkShips:
                        continue
                for hori in [True, False]:
                    if hori == False: # Vert
                        for i in range (10):
                            for j in range (11-ships[s]):
                                if check_crash(self.yourBoard, ships[s], j, i, hori) == False:
                                    for k in range (ships[s]):
                                        if self.yourBoard[j + k][i] != '*': # Should not ideally find '*' anyway
                                            bg[j + k][i] = bg[j + k][i] + 1
                                            if bg[j + k][i] > maxi:
                                                maxi = bg[j + k][i]
                                                max_i = i
                                                max_j = j + k
                    else: # Hori
                        for j in range (10):
                            for i in range (11-ships[s]):
                                if check_crash(self.yourBoard, ships[s], j, i, hori) == False:
                                    for k in range (ships[s]):
                                        if self.yourBoard[j][i + k] != '*': # Should not ideally find '*' anyway
                                            bg[j][i + k] = bg[j][i + k] + 1
                                            if bg[j][i + k] > maxi:
                                                maxi = bg[j][i + k]
                                                max_i = i + k
                                                max_j = j
        return max_j, max_i

    def updateStatus(self, resp, rfire, cfire, smart):
        if resp == '.':
            self.yourBoard[rfire][cfire] = '_'
            #ai2_board[rfire][cfire] = '_'
        elif(resp == '*'):
            self.unresolvedHits.append([rfire, cfire])
            self.yourBoard[rfire][cfire] = '*'
        else:
            self.yourBoard[rfire][cfire] = '*'
            self.unresolvedHits.append([rfire, cfire])
            self.sunkShips[resp] = [rfire, cfire]

            if(not smart):
                self.yourBoard[rfire][cfire] = resp 
            else:
               while(True):
                    for i in self.sunkShips:
                        posi = self.checkShipPoss(i, self.sunkShips[i][0], self.sunkShips[i][1])
                        if(len(posi) == 1):
                            if(posi[0][3]):
                                for j in range(posi[0][2], posi[0][2]+ships[i]):
                                    self.yourBoard[posi[0][1]][j] = i
                                    self.unresolvedHits.remove([posi[0][1], j])
                            else:
                                for j in range(posi[0][1], posi[0][1]+ships[i]):
                                    self.yourBoard[j][posi[0][2]] = i
                                    self.unresolvedHits.remove([j, posi[0][2]])
                            break
                    if(len(posi) != 1):
                        break

            if(len(self.sunkShips) == 5):
                return True

            return False

    def giveResponse(self, r, c):
        # . _ * A B C D S
        if(self.myBoard[r][c] == '.'):
            return self.myBoard[r][c]
        else:
            self.myHits[self.myBoard[r][c]] += 1
            if(ships[self.myBoard[r][c]] == self.myHits[self.myBoard[r][c]]):
                return self.myBoard[r][c]
            else:
                return '*'

    def setBoard(self, board):
        tempBoard = []
        lines = board.split("\n")
        for i in range(len(lines)):
            tempBoard.append(lines[i].split(" ")[-10:])
        self.myBoard = tempBoard
        pprint(self.myBoard)


    def printBoard(self):
        for i in range(10):
            for j in range(10):
                print(self.myBoard[i][j], end = ' ')
            print(" "*10, end = '')
            for j in range(10):
                print(self.yourBoard[i][j], end = ' ')
            print()

class game:
    def __init__(self):
        self.probAi = ai()
        self.rlAi = ai()

        self.turns = 0
    
    def reset(self):
        del self.probAi
        del self.rlAi

        self.probAi = ai()
        self.rlAi = ai()
        self.turns = 0

        self.probAi.placeShips()
        self.rlAi.placeShips()

        return self.convertBoard()
    
    def convertBoard(self):

        characters = {
            '.': 1.0/128,
            '_': 2.0/128,
            '*': 4.0/128,
            'A': 8.0/128,
            'B': 16.0/128,
            'C': 32.0/128,
            'D': 64.0/128,
            'S': 128.0/128
        }

        """
        myboard = []
        for i in range(10):
            line = []
            for j in range(10):
                line.append(characters.index(self.rlAi.myBoard[i][j]))
            myboard.append(line)
        """
        
        yourboard = []
        for i in range(10):
            line = []
            for j in range(10):
                line.append(characters[self.rlAi.yourBoard[i][j]])
            yourboard.append(line)
        
        return yourboard

    def step(self, r, c, oldRew):
        # 5 possibilities for reward/punishment:
        # Hit: 1, Miss: -1, Sink: A:2, B: 3, C: 4, D: 5, S: 4, Win: 17, Lose: -17

        done = False

        self.turns += 1

        resp = self.probAi.giveResponse(r, c)

        #print("RlAi hit:", r, c, resp)

        if(resp == '.'):
            reward = -0.5
        else:
            if(oldRew > 0):
                reward = oldRew * 1.1
            else:
                reward = 1

        if(self.rlAi.updateStatus(resp, r, c, False)):

            """
            reward = 80 - self.turns

            if(reward >= 0):
                reward += 17
            else:
                reward -= 17
            """

            done = True

        yourboard = self.convertBoard()

        return yourboard, done, reward

    def render(self):
        self.rlAi.printBoard()
        #print('-'*10)
        #self.probAi.printBoard()
        print('='*20)

#---------------

class policyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(policyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fcx = nn.Linear(hidden_size, output_size)
        self.fcy = nn.Linear(hidden_size, output_size)

        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')  # Xavier uniform for weights
        init.zeros_(self.fc1.bias)

        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')  # Xavier uniform for weights
        init.zeros_(self.fc2.bias)

        torch.nn.init.kaiming_uniform_(self.fcx.weight, nonlinearity='relu')  # Xavier uniform for weights
        init.zeros_(self.fcx.bias)

        torch.nn.init.kaiming_uniform_(self.fcy.weight, nonlinearity='relu')  # Xavier uniform for weights
        init.zeros_(self.fcy.bias)

        #if(torch.isnan(self.fc1.weight).any()):
        #    pass

    def forward(self, x):
        logits = torch.nn.functional.leaky_relu(self.fc1(x))
        #if(any(math.isnan(x) for x in logits)):
        #    pass
        logits = torch.nn.functional.leaky_relu(self.fc2(logits))
        logitx = torch.nn.functional.leaky_relu(self.fcx(logits))
        logity = torch.nn.functional.leaky_relu(self.fcy(logits))

        xprob = torch.softmax(logitx, dim = -1)
        yprob = torch.softmax(logity, dim = -1)

        return xprob, yprob

"""
def discount_rewards(r):
    # Take 1D float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0:  # reset the sum since this was a game boundary (Pong specific)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def findAction(r, c):
    x = [0]*r
    x.append(1)
    x.append([0]*(9-r))

    y = [0]*c
    y.append(1)
    y.append([0]*(9-c))

    return x, y
"""

def actionTaken(xProb, yProb, maskGrid):
    for i in range(10):
        for j in range(10):
            if(maskGrid[i][j] != 0):
                maskGrid[i][j] = (xProb[i].item()*yProb[j].item())
    
    action = torch.multinomial(torch.tensor(maskGrid).view(-1), 1).item()
    
    r = action // 10
    c = action % 10

    maskGrid[r][c] = 0

    return r, c

def init_training():

    batch_size = 10  # every how many episodes to do a param update?
    gamma = 0.99  # discount factor for reward
    resume = False  # resume from previous checkpoint?
    render = False

    # Model initialization
    model = policyNetwork(input_size=100, hidden_size=200, output_size=10)  # 1 output for probability of action 2
    optimizer = optim.RMSprop(model.parameters(), lr=1e-6, alpha=0.99, eps=1e-7)
    #optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-7)

    env = game()

    lossTot = []
    rewardTot = []
    stepTot = []
    epi = 1

    f = open("./loss.txt", "w")

    writer = SummaryWriter('runs/OnePlayer')

    runningLoss = 0.0

    while(True):
        yrB = env.reset()
        rewards, logProbXs, logProbYs = [], [], []
        done = False
        startTime = currentMilliTime()
        maskGrid = np.ones((10, 10))
        reward = -1
        if(render):
            env.render()
        while not done:

            # Preprocess the observation
            x = torch.tensor(np.vstack(yrB).ravel(), dtype=torch.float32)

            # Forward the policy network and sample an action
            xprob, yprob = model(x)
            
            # log_prob = policy_network(state).log_prob(action)

            #r, c = torch.multinomial(xprob, 1).item(), torch.multinomial(yprob, 1).item()
            r, c = actionTaken(xprob, yprob, maskGrid)

            logProbX = torch.log(xprob[r])
            logProbY = torch.log(yprob[c])

            #x, y = findAction(r, c) # "fake label"

            # Step the environment
            yrB, done, reward = env.step(r, c, reward)
            rewards.append(reward)
            logProbXs.append(logProbX)
            logProbYs.append(logProbY)
            
            #loss = - log_prob * reward
            if(render):
                print(len(rewards))
                env.render()

        
        # Stack together all inputs, action gradients, and rewards for this episode
        # rewards -= np.mean(rewards)
        #print(len(rewards), rewards)

        rewardTot.append(sum(rewards))
        stepTot.append(len(rewards))

        rewards = (rewards - np.mean(rewards))/np.std(rewards)

        epr = torch.tensor(rewards, dtype=torch.float32)

        # TODO: Compute the discounted reward, see Karpathy code

        # Modulate the gradient with advantage
        logProbXs = torch.stack(logProbXs)
        logProbYs = torch.stack(logProbYs)

        lossX = logProbXs * epr * -1
        lossY = logProbYs * epr * -1
        loss = (lossX + lossY).mean()

        lossTot.append(loss.item())

        if(epi % 100 == 0):
            pickle.dump(model, open('model.p', 'wb'))
            writer.add_scalar('loss_100epi', sum(lossTot)/100, epi, currentMilliTime()/1000)
            writer.add_scalar('reward_100epi', sum(rewardTot)/100, epi, currentMilliTime()/1000)
            writer.add_scalar('step_100epi', sum(stepTot)/100, epi, currentMilliTime()/1000)

            f.write("%5d %s, %s, %s\n" % (epi, str(sum(lossTot)/100), str(sum(rewardTot)/100), str(sum(stepTot)/100)))
            f.flush()

            lossTot, rewardTot, stepTot = [], [], []
            #print(epi, loss, currentMilliTime()-startTime)
        
        #print(epi, logProbXs, logProbYs, rewards, loss, currentMilliTime()-startTime)

        #for name, param in model.named_parameters():
        #    print(f"Parameter: {name}, Value: {param.data}")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epi += 1

        #for name, param in model.named_parameters():
        #    print(f"Parameter: {name}, Value: {param.data}")
    

init_training()