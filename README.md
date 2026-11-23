# Huong dan su dung - Bai 1 Tic-Tac-Toe AI

Tai lieu nay danh cho cac thanh vien trong nhom khi chay va kiem thu
chuong trinh [tictactoe.cpp](tictactoe.cpp).

## 1. Yeu cau

- he dieu hanh Linux;
- trinh bien dich C++ ho tro C++17, vi du `g++`.
- terminal co the nhap dong va cot.

## 2. Bien dich

Mo terminal tai thu muc du an:

```bash
cd "/home/overmind/Desktop/C++/AI mid terms"
g++ -std=c++17 -O2 -Wall -Wextra -pedantic tictactoe.cpp -o tictactoe
```

Lenh tren tao file thuc thi `tictactoe`.

## 3. Chay chuong trinh tuong tac

```bash
./tictactoe
```

Sau do:

1. nhap `1` de choi voi AI MCTS;
2. chon ban `3` hoac `5`;
3. chon quan `X` hoac `O`;
4. khi den luot minh, nhap dong va cot, cach nhau mot dau cach.

Vi du:

```text
Enter row column: 2 3
```

Co nghia la dat quan tai dong 2, cot 3.

Luu y: tren ban `5x5`, nguoi hoac may chien thang khi co 5 quan
thang hang ngang, doc, hoac cheo.

## 4. Chay thuc nghiem benchmark

Chay tu menu bang `./tictactoe`, sau do nhap `2`, hoac dung truc tiep:

```bash
./tictactoe bench
```

Ket qua benchmark duoc in theo dinh dang CSV:

```text
board,c,iterations,rollout,win,draw,lose
```

Luu vao file de mo bang Excel/Python:

```bash
./tictactoe bench > results.csv
```

Chuong trinh hien thu nghiem voi:

- ban `3x3` va `5x5`;
- `c` thuoc `{0.5, 1.0, 1.4, 2.0}`;
- iteration thuoc `{100, 500, 1000, 5000}`;
- rollout ngau nhien va heuristic;
- 10 tran cho moi truong hop va doi luot di truoc.

Benchmark co the chay lau. De giam thoi gian khi kiem thu, cac thanh vien
co the giam `iterationsList`, so tran trong `runExperiment`, hoac rang buoc
test trong `runBenchmarks()`; giai phap nop bai nen dung muc nghiem du cua de.

## 5. Cac phan cong kiem nghiem goi y

- Thanh vien 1: kiem tra luat choi, cac nuoc chien thang va hoa.
- Thanh vien 2: kiem tra Minimax + Alpha-Beta va do so nut tim kiem.
- Thanh vien 3: kiem tra MCTS, UCT, `W`, `N`, `W/N`.
- Thanh vien 4: chay benchmark, ve bieu do va rut ra ket luan.
- Thanh vien 5: so sanh random rollout voi heuristic rollout.

## 6. Vi tri cac phan chinh trong ma nguon

- `Board`: luu va kiem tra trang thai ban co.
- `minimax(...)`: thuat toan Minimax co Alpha-Beta pruning.
- `positionalScore(...)`: ham luong gia vi tri.
- `MCTSNode`: node cua cay MCTS.
- `select(...)`: cong thuc UCT.
- `rollout(...)`: mo phong tran dau den ket thuc.
- `heuristicMove(...)`: rollout heuristic bat buoc thoat va chan.
- `runBenchmarks(...)`: tao du lieu CSV cho thuc nghiem.

## 7. Giai thich lua chon nuoc cua MCTS

O nut goc, chuong trinh lua nuoc co `N` lon nhat sau Cac lan iteration.
Day la nuoc da duoc thu thuong xuyen va co uoc luong ro rang hon.
Tuy `W/N` co the rat cao tu mot mau it, `N` lon chon nuoc dang tin cay hon
cho ket qua cuoi cung.
