import sys
import copy
from itertools import permutations
from queue import PriorityQueue
import unicodedata

# Global Variables
Key_list = []
Start = []
End = []
Turn = []
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]


def bfs(m, n, maze):
    global Key_list, Start, End, dx, dy
    ans_cnt = sys.maxsize
    ans_trace = []
    visit_node_cnt = 0

    for turn in range(len(Key_list)):
        start_pos = Start
        key_pos = copy.deepcopy(Key_list[turn])
        key_pos.append(End)
        temp_cnt = 0
        trace = [[0 for col in range(n)] for row in range(m)]

        for target_turn in range(len(key_pos)):
            queue = [start_pos]
            old_start_pos = start_pos
            target_pos = key_pos[target_turn]
            visited = [[0 for col in range(n)] for row in range(m)]
            visited[start_pos[0]][start_pos[1]] = 1
            temp_trace = [[[0, 0] for col in range(n)] for row in range(m)]
            flag = False

            while len(queue) != 0:
                q_size = len(queue)

                while q_size != 0:
                    x, y = queue[0][0], queue[0][1]
                    queue.pop(0)
                    q_size -= 1

                    for i in range(4):
                        nx, ny = x + dx[i], y + dy[i]

                        if 0 <= nx < m and 0 <= ny < n:
                            if maze[nx][ny] == 1:
                                continue
                            if [nx, ny] == target_pos:
                                flag = True
                                start_pos = target_pos
                                temp_trace[nx][ny] = [x, y]
                                break
                            if visited[nx][ny] == 0:
                                visited[nx][ny] = 1
                                queue.append((nx, ny))
                                temp_trace[nx][ny] = [x, y]

                    visit_node_cnt += 1

                temp_cnt += 1

                if flag:
                    while True:
                        trace_x, trace_y = target_pos[0], target_pos[1]
                        trace[trace_x][trace_y] = 1
                        target_pos = temp_trace[trace_x][trace_y]
                        if target_pos == old_start_pos:
                            break
                    break

        if temp_cnt < ans_cnt:
            ans_cnt = temp_cnt
            ans_trace = trace

    print_func(maze, ans_trace, 'BFS', ans_cnt, visit_node_cnt)


def ids(m, n, maze):
    global Key_list, Start, End, dx, dy
    ans_cnt = sys.maxsize
    ans_trace = []
    visit_node_cnt = 0

    for turn in range(len(Key_list)):
        start_pos = Start
        key_pos = copy.deepcopy(Key_list[turn])
        key_pos.append(End)
        temp_cnt = 0
        trace = [[0 for col in range(n)] for row in range(m)]

        for target_turn in range(len(key_pos)):
            target_pos = key_pos[target_turn]
            old_start_pos = start_pos
            d_limit = 0
            is_find = False
            temp_trace = [[[0, 0] for col in range(n)] for row in range(m)]

            while not is_find:
                visited = [[0 for col in range(n)] for row in range(m)]
                stack = [[start_pos, 0]]

                while len(stack) != 0:
                    x, y = stack[-1][0][0], stack[-1][0][1]
                    depth = stack[-1][1]
                    stack.pop()

                    if d_limit <= depth:
                        continue

                    for i in range(4):
                        nx, ny = x + dx[i],  y + dy[i]
                        ndepth = depth + 1

                        if 0 <= nx < m and 0 <= ny < n:
                            if maze[nx][ny] == 1:
                                continue
                            if [nx, ny] == target_pos:
                                is_find = True
                                stack.append([[nx, ny], ndepth])
                                start_pos = target_pos
                                temp_trace[nx][ny] = [x, y]
                                break
                            if visited[nx][ny] == 0:
                                visited[nx][ny] = 1
                                stack.append([[nx, ny], ndepth])
                                temp_trace[nx][ny] = [x, y]

                    visit_node_cnt += 1

                    if is_find:
                        break

                if is_find:
                    temp_cnt += stack[-1][1]
                    while True:
                        trace_x, trace_y = target_pos[0], target_pos[1]
                        trace[trace_x][trace_y] = 1
                        target_pos = temp_trace[trace_x][trace_y]
                        if target_pos == old_start_pos:
                            break
                else:
                    d_limit += 1

        if temp_cnt < ans_cnt:
            ans_cnt = temp_cnt
            ans_trace = trace

    print_func(maze, ans_trace, 'IDS', ans_cnt, visit_node_cnt)


def gbfs(m, n, maze):
    global Key_list, Start, End, dx, dy
    ans_cnt = sys.maxsize
    ans_trace = []
    visit_node_cnt = 0

    for turn in range(len(Key_list)):
        start_pos = Start
        key_pos = copy.deepcopy(Key_list[turn])
        key_pos.append(End)
        temp_cnt = 0
        trace = [[0 for col in range(n)] for row in range(m)]

        for target_turn in range(len(key_pos)):
            prio_q = PriorityQueue()
            prio_q.put([-1, start_pos, 0])
            old_start_pos = start_pos
            target_pos = key_pos[target_turn]
            visited = [[0 for col in range(n)] for row in range(m)]
            visited[start_pos[0]][start_pos[1]] = 1
            temp_trace = [[[0, 0] for col in range(n)] for row in range(m)]
            flag = False

            while not prio_q.empty():
                now = prio_q.get()
                x, y = now[1][0], now[1][1]
                temp_cost = now[2]

                for i in range(4):
                    nx, ny = x + dx[i], y + dy[i]

                    if 0 <= nx < m and 0 <= ny < n:
                        if maze[nx][ny] == 1:
                            continue
                        if [nx, ny] == target_pos:
                            flag = True
                            start_pos = target_pos
                            temp_trace[nx][ny] = [x, y]
                            temp_cnt += temp_cost + 1
                            break
                        if visited[nx][ny] == 0:
                            visited[nx][ny] = 1
                            prio_q.put([heuristic([nx, ny], target_pos), [nx, ny], temp_cost + 1])
                            temp_trace[nx][ny] = [x, y]

                visit_node_cnt += 1

                if flag:
                    while True:
                        trace_x, trace_y = target_pos[0], target_pos[1]
                        trace[trace_x][trace_y] = 1
                        target_pos = temp_trace[trace_x][trace_y]
                        if target_pos == old_start_pos:
                            break
                    break

        if temp_cnt < ans_cnt:
            ans_cnt = temp_cnt
            ans_trace = trace

    print_func(maze, ans_trace, 'GBFS', ans_cnt, visit_node_cnt)


def a_star(m, n, maze):
    global Key_list, Start, End, dx, dy
    ans_cnt = sys.maxsize
    ans_trace = []
    visit_node_cnt = 0

    for turn in range(len(Key_list)):
        start_pos = Start
        key_pos = copy.deepcopy(Key_list[turn])
        key_pos.append(End)
        temp_cnt = 0
        trace = [[0 for col in range(n)] for row in range(m)]

        for target_turn in range(len(key_pos)):
            prio_q = PriorityQueue()
            old_start_pos = start_pos
            target_pos = key_pos[target_turn]
            visited = [[0 for col in range(n)] for row in range(m)]
            visited[start_pos[0]][start_pos[1]] = 1
            temp_trace = [[[0, 0] for col in range(n)] for row in range(m)]
            prio_q.put([-1, start_pos, 0])
            flag = False

            while not prio_q.empty():
                now = prio_q.get()
                x, y = now[1][0], now[1][1]
                temp_cost = now[2]

                for i in range(4):
                    nx, ny = x + dx[i], y + dy[i]

                    if 0 <= nx < m and 0 <= ny < n:
                        if maze[nx][ny] == 1:
                            continue
                        if [nx, ny] == target_pos:
                            flag = True
                            start_pos = target_pos
                            temp_trace[nx][ny] = [x, y]
                            temp_cnt += temp_cost + 1
                            break
                        if visited[nx][ny] == 0:
                            visited[nx][ny] = 1
                            prio_q.put([heuristic([nx, ny], target_pos) + temp_cost + 1, [nx, ny], temp_cost + 1])
                            temp_trace[nx][ny] = [x, y]

                visit_node_cnt += 1

                if flag:
                    while True:
                        trace_x, trace_y = target_pos[0], target_pos[1]
                        trace[trace_x][trace_y] = 1
                        target_pos = temp_trace[trace_x][trace_y]
                        if target_pos == old_start_pos:
                            break
                    break

        if temp_cnt < ans_cnt:
            ans_cnt = temp_cnt
            ans_trace = trace

    print_func(maze, ans_trace, 'A_star', ans_cnt, visit_node_cnt)


def print_func(maze, trace, algorithm_name, ans, visit_node_cnt):
    output = copy.deepcopy(maze)

    for x in range(len(maze)):
        for y in range(len(maze[x])):
            if (maze[x][y] == 2 or maze[x][y] == 6) and trace[x][y] == 1:
                output[x][y] = 5

    file = open('Maze_' + Turn + '_' + algorithm_name + '_output.txt', 'w')
    vstr = ''

    for a in output:
        for b in a:
            vstr = vstr + str(b) + ' '
        vstr += '\n'

    file.writelines(vstr)
    file.write('---\n')
    file.write('length=' + str(ans) + '\n')
    file.write('time=' + str(visit_node_cnt))
    file.close()


def find_info(maze):
    global Key_list, Start, End
    Key_list.clear()
    key_pos = []

    for row, x in enumerate(maze):
        for col, y in enumerate(x):
            if y == 6:
                key_pos.append([row, col])
            elif y == 3:
                Start = [row, col]
            elif y == 4:
                End = [row, col]

    temp = list(permutations(key_pos, len(key_pos)))
    for i in temp:
        Key_list.append(list(i))


def heuristic(start, end):
    return abs(end[0] - start[0]) + abs(end[1] - start[1])


def maze_solving():
    global Turn
    file_names = ['Maze_1.txt', 'Maze_2.txt', 'Maze_3.txt', 'Maze_4.txt']

    for x in file_names:
        my_file = open(x, 'r')
        input_1 = my_file.readline().replace("\n", "").split(' ')
        maze = []
        while True:
            input_2 = my_file.readline().split('\n')[0]
            if input_2 == "":
                break
            temp_maze = []
            for temp in input_2:
                temp_maze.append(int(temp))
            maze.append(temp_maze)

        Turn = input_1[0]
        m = int(input_1[1])
        n = int(input_1[2])

        find_info(maze)

        print('Maze_' + Turn)

        bfs(m, n, maze)
        ids(m, n, maze)
        gbfs(m, n, maze)
        a_star(m, n, maze)

        print()


if __name__ == "__main__":
    maze_solving()
