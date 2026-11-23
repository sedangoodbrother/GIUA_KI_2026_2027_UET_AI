#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace std;

struct Board {
    int size = 0;
    int winLength = 0;
    vector<char> cells;
    int moves = 0;

    Board() = default;
    Board(int n, int k) : size(n), winLength(k), cells(n * n, ' ') {}

    bool inBounds(int r, int c) const {
        return r >= 0 && r < size && c >= 0 && c < size;
    }
    bool validMove(int r, int c) const {
        return inBounds(r, c) && cells[r * size + c] == ' ';
    }
    void playMove(int r, int c, char player) {
        if (validMove(r, c)) {
            cells[r * size + c] = player;
            ++moves;
        }
    }
    void undoMove(int r, int c) {
        if (inBounds(r, c) && cells[r * size + c] != ' ') {
            cells[r * size + c] = ' ';
            --moves;
        }
    }
    bool isFull() const { return moves == size * size; }

    bool wins(char player) const {
        const int dr[4] = {0, 1, 1, 1};
        const int dc[4] = {1, 0, 1, -1};
        for (int r = 0; r < size; ++r) {
            for (int c = 0; c < size; ++c) {
                if (cells[r * size + c] != player) continue;
                for (int d = 0; d < 4; ++d) {
                    bool line = true;
                    for (int k = 0; k < winLength; ++k) {
                        int rr = r + dr[d] * k;
                        int cc = c + dc[d] * k;
                        if (!inBounds(rr, cc) ||
                            cells[rr * size + cc] != player) {
                            line = false;
                            break;
                        }
                    }
                    if (line) return true;
                }
            }
        }
        return false;
    }
    vector<pair<int, int>> emptyCells() const {
        vector<pair<int, int>> result;
        for (int r = 0; r < size; ++r)
            for (int c = 0; c < size; ++c)
                if (cells[r * size + c] == ' ')
                    result.emplace_back(r, c);
        return result;
    }
    void print() const {
        cout << "   ";
        for (int c = 0; c < size; ++c) cout << setw(3) << c + 1;
        cout << '\n';
        for (int r = 0; r < size; ++r) {
            cout << setw(2) << r + 1 << " ";
            for (int c = 0; c < size; ++c)
                cout << setw(3)
                     << (cells[r * size + c] == ' ' ? '.' : cells[r * size + c]);
            cout << '\n';
        }
    }
};

static char otherPlayer(char player) {
    return player == 'X' ? 'O' : 'X';
}

// ========== Minimax + Alpha-Beta ==========
struct MinimaxStats {
    uint64_t nodes = 0;
};

static int evaluateTerminal(const Board& board, char ai, char opponent) {
    if (board.wins(ai)) return 100000;
    if (board.wins(opponent)) return -100000;
    if (board.isFull()) return 0;
    return numeric_limits<int>::min();
}

static int positionalScore(const Board& board, char player) {
    int score = 0;
    char opponent = otherPlayer(player);
    const int dr[4] = {0, 1, 1, 1};
    const int dc[4] = {1, 0, 1, -1};
    for (int r = 0; r < board.size; ++r) {
        for (int c = 0; c < board.size; ++c) {
            if (board.cells[r * board.size + c] != ' ') continue;
            for (int d = 0; d < 4; ++d) {
                int own = 0, enemy = 0;
                for (int k = -(board.winLength - 1); k < board.winLength; ++k) {
                    int rr = r + dr[d] * k;
                    int cc = c + dc[d] * k;
                    if (!board.inBounds(rr, cc)) continue;
                    char v = board.cells[rr * board.size + cc];
                    if (v == player) ++own;
                    if (v == opponent) ++enemy;
                }
                if (enemy == 0) score += own * own;
                if (own == 0) score -= enemy * enemy;
            }
        }
    }
    return score;
}

static int minimax(Board& board, int depth, int alpha, int beta,
                   bool maximizing, char ai, char opponent,
                   MinimaxStats& stats) {
    ++stats.nodes;
    int terminal = evaluateTerminal(board, ai, opponent);
    if (terminal != numeric_limits<int>::min()) return terminal - depth;
    if (depth == 0) return positionalScore(board, ai);

    char player = maximizing ? ai : opponent;
    for (const auto& mv : board.emptyCells()) {
        board.playMove(mv.first, mv.second, player);
        int value = minimax(board, depth - 1, alpha, beta,
                            !maximizing, ai, opponent, stats);
        board.undoMove(mv.first, mv.second);
        if (maximizing) {
            if (value > alpha) alpha = value;
        } else {
            if (value < beta) beta = value;
        }
        if (beta <= alpha) break;
    }
    return maximizing ? alpha : beta;
}

struct SearchResult {
    pair<int, int> move{-1, -1};
    MinimaxStats stats;
};

SearchResult findBestMinimaxMove(Board& board, char ai, int depth) {
    char opponent = otherPlayer(ai);
    SearchResult result;
    int alpha = numeric_limits<int>::min();
    int beta = numeric_limits<int>::max();
    int best = alpha;

    for (const auto& mv : board.emptyCells()) {
        board.playMove(mv.first, mv.second, ai);
        int value = minimax(board, depth - 1, alpha, beta,
                            false, ai, opponent, result.stats);
        board.undoMove(mv.first, mv.second);
        if (value > best) {
            best = value;
            result.move = mv;
        }
        alpha = max(alpha, best);
    }
    return result;
}

// ========== MCTS + UCT ==========
struct MCTSNode {
    Board board;
    MCTSNode* parent = nullptr;
    vector<MCTSNode*> children;
    vector<pair<int, int>> untriedMoves;
    pair<int, int> move{-1, -1};
    char playerToMove;
    char rootPlayer;
    int visits = 0;
    double reward = 0.0;

    MCTSNode(Board b, MCTSNode* p, pair<int, int> m,
             char nextPlayer, char root)
        : board(b), parent(p), move(m),
          playerToMove(nextPlayer), rootPlayer(root) {
        untriedMoves = board.emptyCells();
    }
    ~MCTSNode() {
        for (MCTSNode* child : children) delete child;
    }
    bool terminal() const {
        return board.wins('X') || board.wins('O') || board.isFull();
    }

    MCTSNode* select(double c) const {
        MCTSNode* best = nullptr;
        double bestScore = -numeric_limits<double>::infinity();
        for (MCTSNode* child : children) {
            double mean = child->visits ? child->reward / child->visits : 0.0;
            if (playerToMove != rootPlayer) mean = -mean;
            double explore = c * sqrt(log(max(1, visits)) /
                                      max(1, child->visits));
            double score = mean + explore;
            if (score > bestScore) {
                bestScore = score;
                best = child;
            }
        }
        return best;
    }

    MCTSNode* expand(mt19937& rng) {
        if (untriedMoves.empty()) return nullptr;
        uniform_int_distribution<size_t> dist(0, untriedMoves.size() - 1);
        size_t index = dist(rng);
        pair<int, int> mv = untriedMoves[index];
        untriedMoves.erase(untriedMoves.begin() + index);
        Board childBoard = board;
        childBoard.playMove(mv.first, mv.second, playerToMove);
        MCTSNode* child = new MCTSNode(childBoard, this, mv,
                                       otherPlayer(playerToMove), rootPlayer);
        children.push_back(child);
        return child;
    }
    void backPropagate(double result) {
        MCTSNode* node = this;
        while (node) {
            ++node->visits;
            node->reward += result;
            node = node->parent;
        }
    }
};

static pair<int, int> heuristicMove(Board& board, char player) {
    vector<pair<int, int>> moves = board.emptyCells();
    if (moves.empty()) return {-1, -1};
    char opponent = otherPlayer(player);

    for (const auto& mv : moves) {
        board.playMove(mv.first, mv.second, player);
        bool win = board.wins(player);
        board.undoMove(mv.first, mv.second);
        if (win) return mv;
    }
    for (const auto& mv : moves) {
        board.playMove(mv.first, mv.second, opponent);
        bool win = board.wins(opponent);
        board.undoMove(mv.first, mv.second);
        if (win) return mv;
    }

    int center = board.size / 2;
    pair<int, int> best = moves.front();
    int bestDistance = numeric_limits<int>::max();
    for (const auto& mv : moves) {
        int distance = abs(mv.first - center) + abs(mv.second - center);
        if (distance < bestDistance) {
            bestDistance = distance;
            best = mv;
        }
    }
    return best;
}

static double rollout(Board board, char playerToMove, char rootPlayer,
                      bool heuristic, mt19937& rng) {
    while (!board.wins('X') && !board.wins('O') && !board.isFull()) {
        pair<int, int> mv;
        if (heuristic) {
            mv = heuristicMove(board, playerToMove);
        } else {
            vector<pair<int, int>> moves = board.emptyCells();
            uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            mv = moves[dist(rng)];
        }
        board.playMove(mv.first, mv.second, playerToMove);
        playerToMove = otherPlayer(playerToMove);
    }
    if (board.wins(rootPlayer)) return 1.0;
    if (board.wins(otherPlayer(rootPlayer))) return -1.0;
    return 0.0;
}

pair<int, int> findBestMCTSMove(Board board, char rootPlayer,
                                int iterations, double c,
                                bool heuristic, mt19937& rng) {
    MCTSNode root(board, nullptr, {-1, -1}, rootPlayer, rootPlayer);
    for (int i = 0; i < iterations; ++i) {
        MCTSNode* node = &root;
        while (!node->terminal() && node->untriedMoves.empty()) {
            node = node->select(c);
            if (!node) break;
        }
        if (node && !node->terminal() && !node->untriedMoves.empty())
            node = node->expand(rng);

        double result = rollout(node ? node->board : board,
                                node ? node->playerToMove : rootPlayer,
                                rootPlayer, heuristic, rng);
        if (node) node->backPropagate(result);
        else root.backPropagate(result);
    }

    MCTSNode* best = nullptr;
    int bestVisits = -1;
    for (MCTSNode* child : root.children) {
        if (child->visits > bestVisits) {
            bestVisits = child->visits;
            best = child;
        }
    }
    if (best) return best->move;
    return board.emptyCells().front();
}

static void printRootMCTSStats(Board board, char rootPlayer,
                               int iterations, double c,
                               bool heuristic, mt19937& rng) {
    MCTSNode root(board, nullptr, {-1, -1}, rootPlayer, rootPlayer);
    for (int i = 0; i < iterations; ++i) {
        MCTSNode* node = &root;
        while (!node->terminal() && node->untriedMoves.empty()) {
            node = node->select(c);
            if (!node) break;
        }
        if (node && !node->terminal() && !node->untriedMoves.empty())
            node = node->expand(rng);
        double result = rollout(node ? node->board : board,
                                node ? node->playerToMove : rootPlayer,
                                rootPlayer, heuristic, rng);
        if (node) node->backPropagate(result);
        else root.backPropagate(result);
    }

    cout << "\nRoot MCTS statistics (N vs W/N):\n";
    cout << "move | W | N | W/N\n";
    for (MCTSNode* child : root.children) {
        double winRate = child->visits
            ? (child->reward / child->visits + 1.0) / 2.0
            : 0.0;
        cout << '(' << child->move.first + 1 << ','
             << child->move.second + 1 << ") | "
             << fixed << setprecision(2) << child->reward
             << " | " << child->visits
             << " | " << winRate << '\n';
    }
}

struct ExperimentResult {
    int win = 0, draw = 0, lose = 0;
};

ExperimentResult runExperiment(int size, int winLength, double c,
                               int iterations, int games,
                               bool heuristicRollout,
                               bool mctsGoesFirst,
                               mt19937& rng) {
    ExperimentResult result;
    for (int game = 0; game < games; ++game) {
        Board board(size, winLength);
        char mctsPlayer = mctsGoesFirst ? 'X' : 'O';
        char player = 'X';
        while (!board.wins('X') && !board.wins('O') && !board.isFull()) {
            pair<int, int> mv;
            if (player == mctsPlayer)
                mv = findBestMCTSMove(board, mctsPlayer, iterations,
                                      c, heuristicRollout, rng);
            else
                mv = heuristicMove(board, player);
            board.playMove(mv.first, mv.second, player);
            player = otherPlayer(player);
        }
        if (board.wins(mctsPlayer)) ++result.win;
        else if (board.wins(otherPlayer(mctsPlayer))) ++result.lose;
        else ++result.draw;
    }
    return result;
}

static void runBenchmarks() {
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    cout << "board,c,iterations,rollout,win,draw,lose\n";
    const vector<double> cs = {0.5, 1.0, 1.4, 2.0};
    const vector<int> iterationsList = {100, 500, 1000, 5000};
    for (int size : {3, 5}) {
        int winLength = size == 3 ? 3 : 5;
        for (double c : cs) {
            for (int iterations : iterationsList) {
                for (bool heuristic : {false, true}) {
                    for (bool mctsFirst : {true, false}) {
                        ExperimentResult r = runExperiment(
                            size, winLength, c, iterations, 10,
                            heuristic, mctsFirst, rng);
                        cout << size << 'x' << size << ',' << c << ','
                             << iterations << ','
                             << (heuristic ? "heuristic" : "random") << ','
                             << r.win << ',' << r.draw << ',' << r.lose << '\n';
                    }
                }
            }
        }
    }
}

static void playHumanVsAI() {
    int size;
    cout << "Choose board size (3 or 5): ";
    cin >> size;
    if (size != 3 && size != 5) {
        cout << "Invalid size.\n";
        return;
    }
    int winLength = size == 3 ? 3 : 5;
    Board board(size, winLength);
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

    cout << "Choose symbol: X (go first) or O: ";
    char human;
    cin >> human;
    human = toupper(human);
    if (human != 'X' && human != 'O') human = 'X';
    char ai = otherPlayer(human);
    char player = 'X';

    while (!board.wins('X') && !board.wins('O') && !board.isFull()) {
        board.print();
        pair<int, int> mv;
        if (player == human) {
            while (true) {
                cout << "Enter row column: ";
                int r, c;
                if (!(cin >> r >> c)) return;
                --r;
                --c;
                if (board.validMove(r, c)) {
                    mv = {r, c};
                    break;
                }
                cout << "Invalid move.\n";
            }
        } else {
            mv = findBestMCTSMove(board, ai, 1000, 1.4, true, rng);
            cout << "AI plays (" << mv.first + 1 << ", "
                 << mv.second + 1 << ")\n";
            printRootMCTSStats(board, ai, 1000, 1.4, true, rng);
        }
        board.playMove(mv.first, mv.second, player);
        player = otherPlayer(player);
    }
    board.print();
    if (board.wins(human)) cout << "You win!\n";
    else if (board.wins(ai)) cout << "AI wins.\n";
    else cout << "Draw.\n";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (argc > 1 && string(argv[1]) == "bench") {
        runBenchmarks();
        return 0;
    }
    cout << "Tic-Tac-Toe (AI assignment)\n";
    cout << "1. Play against MCTS\n";
    cout << "2. Run benchmark experiments\n";
    cout << "Choose: ";
    int choice;
    cin >> choice;
    if (choice == 2) runBenchmarks();
    else playHumanVsAI();
    return 0;
}
