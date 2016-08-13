import java.util.*;

public class NeuralSimulation {

    private static ArrayList<double[]> data;

    public static void main(String[] args) {
//

        ArrayList<double[]> data = new ArrayList<double[]>();
//        double[] d = {1.0,0.0,1.0};
//       // data.add(d);
//        double[] f = {0.0,1.0,1.0};
//        //data.add(f);
//        double[] g = {0.0,0.0,0.0};
//        //data.add(g);
//        //double[] h = {0.1,0.05,0.01,0.99};
//        double[] h = {1.0,-1.0,0.0,0.0,1.0,0.0};
//        data.add(h);
//        double[] j = {0.0,4.0,-1.0,1.0,0.0,0.0};
//        data.add(j);
//        Neural neural = new Neural(3, 3, 3, data, 10000);

//        double[] d = {1.0,0.0,1.0};
//        data.add(d);
//        double[] f = {0.0,1.0,1.0};
//        data.add(f);
//        double[] g = {0.0,0.0,0.0};
//        data.add(g);
//        double[] h = {1.0,1.0,0.0};
//        data.add(h);
        //data= generateTicTacToeData();
        //RBFNeural neural = new RBFNeural(2, 1, data);

        data = generateAddData();

        //data = generateTicTacToeData();
        Neural neural = new Neural(2,5,5,data,10000);
        //Neural neural = new Neural(9, 6, 9, data, 3000);
        //playGame(neural);

    }
    public static ArrayList<double[]> generateAddData()
    {
        ArrayList<double[]> dataTemp = new ArrayList<double[]>();
        for(int x = 0; x < 16; x++){
            for(int y = 0; y < 16; y++){
                double[] d = new double[7];
                d[0] = x;
                d[1] = y;
                String converted = Integer.toBinaryString(x+y);
                for(int q = 0; q < converted.toCharArray().length; q++)
                    d[2+q] = Character.getNumericValue(converted.toCharArray()[q]);
                dataTemp.add(d);
            }
        }
        return dataTemp;
    }
    public static boolean fullGame(double[] b){
        for(double x : b)
            if(x==0.0)
                return false;
        return true;
    }
    public static void playGame(Neural n){
        double[] board = new double[9];
        displayBoard(board);
        int counter = 0;
        while(true){
            if(fullGame(board)){
                counter = 0;
                board = new double[9];
                displayBoard(board);
            }
            if(counter%2==0){
                Scanner infile = new Scanner(System.in);
                int input = infile.nextInt();
                if(board[input]!=0.0)
                    System.out.println("bad move");
                else {
                    board[input] = -1;
                    counter += 1;
                }
            }else{
                double[] sigs = n.calculateSigmoid(board);

                while(true){
                    int maxI = 0;
                    double maxV = 0;
                    for(int x = 0; x < sigs.length; x++){
                        if(sigs[x] > maxV){
                            maxV = sigs[x];
                            maxI = x;
                        }
                    }
                    if(board[maxI]==0) {
                        System.out.println(maxV);
                        board[maxI] = 1.0;
                        counter+=1;
                        break;
                    }else{
                        sigs[maxI] = 0;
                    }
                }
            }
            displayBoard(board);

        }
    }
    public static void displayBoard(double[] board){
        for(int x = 0; x < board.length; x++) {
            if (x == 3 || x == 6)
                System.out.println();
            if(board[x]==-1.0)
                System.out.print("O ");
            else if(board[x]==1.0)
                System.out.print("X ");
            else
                System.out.print("- ");
        }
        System.out.println();
        System.out.println();
    }
    public static ArrayList<double[]> generateTicTacToeData(){
        double[][] a1 = {
                {0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, -1, 0, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, -1, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                {-1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                {0, 0, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                {1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                {0, 0, -1, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                {-1, 0, 0, 1, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                {-1, 1, -1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                {0, 0, 0, -1, 0, 1, 1, -1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                {-1, 1, 0, 0, 0, -1, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, -1, 1, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                {0, 0, -1, 1, 0, -1, 0, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}
        };
        ArrayList<double[]> d = new ArrayList<double[]>();
        for(double[] a : a1) {
            d.add(a);
        }

        for(double[] a : a1){
            int counter = 0;
            double[] b = Arrays.copyOfRange(a, 0, 18);
            while(counter<3) {
                double[] front = Arrays.copyOfRange(b, 0, 9);
                double[] back = Arrays.copyOfRange(b, 9, 18);
                double[] fi = new double[18];
                fi[0] = front[2];
                fi[3] = front[1];
                fi[6] = front[0];
                fi[1] = front[5];
                fi[4] = front[4];
                fi[7] = front[3];
                fi[2] = front[8];
                fi[5] = front[7];
                fi[8] = front[6];

                fi[0 + 9] = back[2];
                fi[3 + 9] = back[1];
                fi[6 + 9] = back[0];
                fi[1 + 9] = back[5];
                fi[4 + 9] = back[4];
                fi[7 + 9] = back[3];
                fi[2 + 9] = back[8];
                fi[5 + 9] = back[7];
                fi[8 + 9] = back[6];
                d.add(fi);
                b = Arrays.copyOfRange(fi,0,18);
                counter += 1;
            }
        }

        return d;
    }
}
