import trueskill
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import random
import itertools
import math


def simulate():
    true_skill = trueskill.TrueSkill(mu=25.0, sigma=8.333333333333334, beta=4.166666666666667, tau=0.08333333333333334,
                                     draw_probability=0.1, backend=None)
    rating_group = []
    rating_group.append((true_skill.create_rating(), true_skill.create_rating()))
    rating_group.append((true_skill.create_rating(), true_skill.create_rating()))
    # assume win rate of p1 = 0.7
    count = 0
    ranks_p1_win = [0, 1]
    ranks_p2_win = [1, 0]
    res = []

    while count < 1000:
        num = random.randint(1, 10)
        if num <= 1:  # adjust winning rate of the match and the
            res.append(0)
        else:
            res.append(1)
        count += 1

    for result in res:
        if result == 1:  # adjust win probability to change fairness of the match after updating skill ratings
            rating_group = true_skill.rate(rating_group, ranks_p1_win)
        else:
            rating_group = true_skill.rate(rating_group, ranks_p2_win)

    print("win probability of the player1:", win_probability(rating_group[0], rating_group[1]))
    print("fairness of the match: ", true_skill.quality(rating_group))  # indication of fairness
    print("skill ratings:")
    print(rating_group[0])
    print(rating_group[1])

def group_test():
    true_skill = trueskill.TrueSkill(mu=25.0, sigma=8.333333333333334, beta=4.166666666666667, tau=0.08333333333333334,
                                     draw_probability=0.1, backend=None)
    # team1 = (true_skill.create_rating(mu=27.35, sigma=8.333333333333334), true_skill.create_rating(mu=27.35, sigma=8.333333333333334))
    # team2 = (true_skill.create_rating(), true_skill.create_rating())
    # team3 = (true_skill.create_rating(mu=22.45, sigma=8.333333333333334), true_skill.create_rating(mu=22.45, sigma=8.333333333333334))
    # team4 = (true_skill.create_rating(mu=19.5, sigma=8.333333333333334), true_skill.create_rating(mu=19.5, sigma=8.333333333333334))

    team1 = (true_skill.create_rating(), true_skill.create_rating())
    team2 = (true_skill.create_rating(), true_skill.create_rating())
    team3 = (true_skill.create_rating(), true_skill.create_rating())
    team4 = (true_skill.create_rating(), true_skill.create_rating())
    # initialize 4 teams with winning probabilty shown in the table

    # rating_group_1_2 = []
    # rating_group_1_2.append(team1)
    # rating_group_1_2.append(team2)
    # print("rating_group_1_2:")
    # print(win_probability(rating_group_1_2[0], rating_group_1_2[1]))
    # print(rating_group_1_2)

    teams = [team1, team2, team3, team4]
    count = 0
    ranks_first_win = [0, 1]
    ranks_second_win = [1, 0]

    prob_table = [[0, 0.6, 0.7, 0.8], [0.4, 0, 0.6, 0.7], [0.3, 0.4, 0, 0.6], [0.2, 0.3, 0.4, 0]]
    # print(teams)
    rank_dif = []
    prob_table_dif = []

    while count < 1000:
        cur_prob_table = []
        num1 = random.randint(1, 4)
        num2 = random.randint(1, 4)
        while num1 == num2:
            num2 = random.randint(1, 4)
        t1 = teams[num1 - 1]
        t2 = teams[num2 - 1]
        res = [t1, t2]
        win_prob = prob_table[num1 - 1][num2 - 1]
        rand = random.random()

        if rand <= win_prob:
            res = true_skill.rate(res, ranks_first_win)
        else:
            res = true_skill.rate(res, ranks_second_win)
        # print(res)
        teams[num1 - 1] = res[0]
        teams[num2 - 1] = res[1]
        original_teams = []
        for team in teams:
            original_teams.append(team)
        cur_prob_table = calculate_cur_prob_table(teams)
        # print(cur_prob_table)
        prob_table_dif.append(calculate_prob_table_dif(cur_prob_table, prob_table))
        # print(original_teams)
        teams.sort(key=get_mu, reverse=True)
        # print(teams)
        rank_dif.append(get_rank_dif(teams, original_teams))
        # teams = original_teams
        count += 1

    print(rank_dif)
    print(prob_table_dif)
    x_axis = []
    index = 0
    while index < len(rank_dif):
        x_axis.append(index)
        index += 1
    # plt.plot(x_axis, rank_dif)

    plt.plot(x_axis, prob_table_dif)
    plt.show()


def calculate_prob_table_dif(table1, table2):
    res = 0
    i = 0
    while i < len(table1):
        j = 0
        while j < len(table1[0]):
            res += abs(table1[i][j] - table2[i][j])
            j += 1
        i += 1
    return res


def calculate_cur_prob_table(teams):
    res = []
    i = 0
    while i < len(teams):
        temp = []
        j = 0
        while j < len(teams):
            if i == j:
                temp.append(0)
            else:
                temp.append(win_probability(teams[i], teams[j]))
            j += 1
        res.append(temp)
        i += 1
    return res


def get_rank_dif(teams1, teams2):
    count = 0
    index = 0
    while index < len(teams1):
        if teams1[index] != teams2[index]:
            count += 1
        index += 1
    return count


def get_mu(team):
    return team[0].mu


def plot(rating):
    x_min = 0.0
    x_max = 50

    mean = rating.mu
    std = rating.sigma

    x = np.linspace(x_min, x_max, 100)
    y = scipy.stats.norm.pdf(x, mean, std)
    plt.plot(x, y, color='coral')
    plt.grid()

    plt.xlim(x_min, x_max)
    plt.ylim(0, 0.25)

    plt.title('How to plot a normal distribution in python with matplotlib', fontsize=10)

    plt.xlabel('x')
    plt.ylabel('Normal Distribution')

    plt.savefig("normal_distribution.png")
    plt.show()

# citation of the method Lee, Heungsub. “TrueSkill¶.” TrueSkill, trueskill.org/.
def win_probability(team1, team2):
    # from paper
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)


if __name__ == '__main__':
    simulate()
    # group_test()
