import java.util.ArrayList;
import java.util.Map;
import java.util.Random;

/**
 * A general feed-forward neural network with one hidden layer
 * Note: There must be at least one hidden layer (if linearly separable,
 * use one node in the hidden layer)
 *
 * Generally, an ideal neural network contains one input/hidden/output layer,
 * with the number of hidden nodes depending on scenario
 *
 * Data must be given the following way:
 * 1. ArrayList containing N data sets
 * 2. Each data set is represented by a double array of following characteristic:
 *      a. First N(inp) items are input data
 *      b. Remaining N(out) items are output data
 *      c. i.e: INPUTS: 1, 0; OUTPUTS: 1
 *              double[] data = {1,0,1}
 *
 * Depending on the given data set, number of inputs/outputs/hidden nodes, and
 * other factors (such as the learning rate/implementing a stochastic approach
 * within backpropagation), the performance of the neural network may vary. The
 * given values work well for the XOR data set. Stochastic approach may be
 * feasible for larger data sets, although not yet implemented
 */

public class Neural {

    private static int input,hidden,output;
    private static ArrayList<double[]> data;
    private static ArrayList<double[]> weights;

    public Neural(int inp, int hid, int out, ArrayList<double[]> d, int epoch){
        input = inp;
        hidden = hid;
        output = out;
        data = d;
        instantiateWeights();

        /* example of weights for testing purposes; ignore
        weights = new ArrayList<double[]>();
        double[] w = {0.35,0.15,0.20};
        weights.add(w);
        double[] w1 = {0.35,0.25,0.30};
        weights.add(w1);
        double[] w2 = {0.60,0.40,0.45};
        weights.add(w2);
        double[] w3 = {0.60,0.50,0.55};
        weights.add(w3);
         */

        backProp(epoch);
    }
    public static void printWeights(){
        System.out.println("Weights: ");
        System.out.print("Hidden: ");
        for(int x = 0; x < hidden; x++) {
            System.out.print("[");
            for (double y : weights.get(x))
                System.out.print(y + ", ");
            System.out.print("]");
        }
        System.out.println();
        System.out.print("Output: ");
        for(int x = hidden; x < output+hidden; x++) {
            System.out.print("[");
            for (double y : weights.get(x))
                System.out.print(y + ", ");
            System.out.print("]");
        }
        System.out.println();
    }
    public static void printArbWeights(ArrayList<double[]> we){
        System.out.println("Chosen Weights: ");
        System.out.print("Hidden: ");
        for(int x = 0; x < hidden; x++) {
            System.out.print("[");
            for (double y : we.get(x))
                System.out.print(y + ", ");
            System.out.print("]");
        }
        System.out.println();
        System.out.print("Output: ");
        for(int x = hidden; x < output+hidden; x++) {
            System.out.print("[");
            for (double y : we.get(x))
                System.out.print(y + ", ");
            System.out.print("]");
        }
        System.out.println();
        System.out.println();
    }
    public static void displayError(double[] errors){
        System.out.println("Error: ");
        for(int x= 0; x < errors.length; x++)
            System.out.println("Data " + x + ": " + errors[x]);
        System.out.println();
    }
    public static void displayOutput(ArrayList<double[]> data){
        System.out.println("Output: ");
        int counter = 0;
        for(double[] d : data){
            double[] sigmoid = calculateSigmoid(d);
            System.out.print("Data " + counter + ": [");
            for(int x = 0; x < sigmoid.length; x++)
                if(x==sigmoid.length-1)
                    System.out.print(sigmoid[x] + "");
                else
                    System.out.print( sigmoid[x] + ", ");
            System.out.println("]");
            counter+=1;
        }
        System.out.println();
    }
    public static void instantiateWeights(){
        weights = new ArrayList<double[]>();
        //double bias = (Math.random());
        //double max = 1/Math.pow(input, 0.5);
        double max = 1.0;
        double bias = new Random().nextDouble()*max*2 - max;
        //weights for hidden layer
        for(int hidCount = 0; hidCount < hidden; hidCount++){
            double[] weight = new double[input+1];
            weight[0] = bias;
            for(int numWeights = 1; numWeights < input+1; numWeights++)
                weight[numWeights] = new Random().nextDouble()*max*2 - max;
            weights.add(weight);
        }
        //weights for output layer
        bias = new Random().nextDouble()*max*2 - max;
        for(int outCount = 0; outCount < output; outCount++){
            double[] weight = new double[hidden+1];
            weight[0] = bias;
            if(hidden==0) {
                weight = new double[input+1];
                weight[0] = bias;
                for(int numWeights = 1; numWeights < input+1; numWeights++)
                    weight[numWeights] = new Random().nextDouble()*max*2 - max;
            }else{
                for(int numWeights = 1; numWeights < hidden+1; numWeights++)
                    weight[numWeights] = new Random().nextDouble()*max*2 - max;
            }
            weights.add(weight);
        }
    }
    public static double calculateY(double[] d, double[] weight){
        double y = 0;
        for(int x = 0; x < weight.length; x++){
            if(x==0)
                y += weight[x];
            else
                y += weight[x]*d[x-1];
        }
        return y;
    }
    public static double[] calculateSigmoid(double[] d){

        //calculate hidden sigmoids first
        double[] sigmoids = new double[hidden];
        for(int hiddenCount = 0; hiddenCount < hidden; hiddenCount++){
            double y = calculateY(d, weights.get(hiddenCount));
            sigmoids[hiddenCount] = 1/(1+Math.pow(Math.E,(-1)*y));
        }

        //calculate final z
        double[] outputSigmoids = new double[output];
        for(int outputCount = hidden; outputCount < output+hidden; outputCount++){
            double y = 0;
            if(hidden==0)
                y = calculateY(d,weights.get(outputCount));
            else
                y = calculateY(sigmoids,weights.get(outputCount));
            outputSigmoids[outputCount-hidden] = 1/(1+Math.pow(Math.E,(-1)*y));
        }
        return outputSigmoids;
    }
    public static double[] calculatePartialSigmoid(double[] d){
        //calculate hidden sigmoids - essentially partialSigmoids (used for backprop)
        double[] sigmoids = new double[hidden];
        for(int hiddenCount = 0; hiddenCount < hidden; hiddenCount++){
            double y = calculateY(d, weights.get(hiddenCount));
            sigmoids[hiddenCount] = 1/(1+Math.pow(Math.E,(-1)*y));
        }
        return sigmoids;
    }
    public static double[] calculateError(){
        double[] errors = new double[data.size()];
        int counter = 0;
        for(double[] x : data){
            double error = 0;
            double[] sigmoids = calculateSigmoid(x);
            for(int sig = 0; sig < sigmoids.length; sig++)
                error += Math.pow(sigmoids[sig]-x[sig+input],2);
            errors[counter] = error;
            counter += 1;
        }
        for(int y = 0; y < errors.length; y++)
            errors[y] = errors[y] * 0.5;
        return errors;
    }
    public ArrayList<double[]> copyWeights(){
        ArrayList<double[]> cop = new ArrayList<double[]>();
        for(double[] d : weights){
            double[] ne = new double[d.length];
            System.arraycopy(d, 0, ne, 0, ne.length);
            cop.add(ne);
        }
        return cop;
    }
    public static void updateWeights(ArrayList<double[]> cp){
        for(int x = 0; x < cp.size(); x++){
            double[] cop = new double[cp.get(x).length];
            System.arraycopy(cp.get(x), 0, cop, 0, cop.length);
            weights.set(x, cop);
        }
    }
    public void backProp(int epoch){
        int counter = 0;
        double lrn = 0.29; //still hard to determine learning rate
        double momentum = 0.9;

        //creating arraylist for prevweightchanges for momentum-based learning
        ArrayList<double[]> prevWeightChanges = new ArrayList<double[]>();
        for(int x = 0; x < hidden; x++) {
            double[] asdf = new double[1+input];
            prevWeightChanges.add(asdf);
        }
        for(int y = 0; y < output; y++) {
            double[] asdf = new double[1+hidden];
            prevWeightChanges.add(asdf);
        }

        while(counter < epoch){  //true portion allows for more accurate weights but does not terminate

            ArrayList<double[]> newWeights = copyWeights();
            for(int weightCounter = 0; weightCounter < hidden*(1+input) + output*(1+hidden); weightCounter += 1){
                if(weightCounter < hidden*(1+input)){
                    //hidden layer weights
                    int positionInArray = weightCounter/(input+1);
                    int positionToCheckBias = weightCounter%(input+1);
                    double dedw = 0;
                    for(double[] d : data){
                        double[] sigmoids = calculateSigmoid(d);
                        int sigmoidCounter = 0;
                        for(double z : sigmoids){
                            double o = d[input+sigmoidCounter];
                            double w = weights.get(hidden+sigmoidCounter)[positionInArray+1];
                            double[] partialSig = calculatePartialSigmoid(d);
                            double z2 = partialSig[positionInArray];

                            if(positionToCheckBias!=0){
                                double inp = d[positionToCheckBias-1];
                                dedw += (z-o)*z*(1-z)*w*z2*(1-z2)*inp;
                            }else{
                                int h = 1;
                                dedw += (z-o)*z*(1-z)*w*z2*(1-z2)*1;
                            }

                            sigmoidCounter++;
                        }
                    }

                    newWeights.get(positionInArray)[positionToCheckBias] = newWeights.get(positionInArray)[positionToCheckBias] - dedw*lrn + momentum*prevWeightChanges.get(positionInArray)[positionToCheckBias];
                    prevWeightChanges.get(positionInArray)[positionToCheckBias] = (-1)*dedw*lrn;

                }else if(weightCounter < hidden*(1+input) + output*(1+hidden)){
                    //output layer weights
                    double dedw = 0.0;
                    double z = 0.0;
                    double o = 0.0;
                    double w = 0.0;
                    int positionInArray = (weightCounter-hidden*(1+input))/(hidden+1);
                    int positionToCheckBias = (weightCounter-hidden*(1+input))%(hidden+1);
                    for(double[] d : data){
                        z = calculateSigmoid(d)[positionInArray];
                        o = d[input+positionInArray];
                        if(positionToCheckBias!=0){
                            double[] partialSig = calculatePartialSigmoid(d);
                            w = partialSig[positionToCheckBias-1];
                            dedw += (z-o)*z*(1-z)*w;
                        }else {  //bias
                            int h = 1;
                            dedw += (z-o)*z*(1-z)*1;
                        }
                    }
                    newWeights.get(hidden+positionInArray)[positionToCheckBias] = newWeights.get(hidden+positionInArray)[positionToCheckBias] - dedw*lrn + momentum*prevWeightChanges.get(hidden+positionInArray)[positionToCheckBias];
                    prevWeightChanges.get(hidden+positionInArray)[positionToCheckBias] = (-1)*dedw*lrn;
                }
            }
            updateWeights(newWeights);
            counter += 1;
            System.out.println(counter);
            displayError(calculateError());
            //displayOutput(data);
        }
        displayOutput(data);
    }
    public static ArrayList<double[]> getWeights(){
        return weights;
    }
}
