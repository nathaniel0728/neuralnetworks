package us.ihmc.exampleSimulations.centeredRod;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by ShadyLady on 7/19/2016.
 */
public class RBFNeural {

    private static ArrayList<double[]> inputs;
    private static ArrayList<double[]> outputs;
    private static int input,output;

    private static ArrayList<double[]> rbf;
    private static ArrayList<double[]> weights;
    private static ArrayList<Double> beta;

    public RBFNeural(int inp, int out, ArrayList<double[]> d){
        input = inp;
        output = out;
        sortData(d);
        createNeuralNetwork();
        evaluateNetworkAndPrint();
    }
    public static void evaluateNetworkAndPrint(){
        for(int y = 0; y < inputs.size(); y++) {
            for (double x : evaluateNetwork(inputs.get(y)))
                System.out.print(x + " ");
            System.out.println();
        }
    }
    public static double activationFunction(double[] data, double[] hid, double b){
        double euclidDist = calculateEuclidDistance(data, hid);
        return Math.pow(Math.E, (-1)*b*Math.pow(euclidDist,2));
    }
    public static double[] evaluateNetwork(double[] data){
        double[] o = new double[output];

        double[] hiddenEval = new double[rbf.size()];
        for(int x = 0; x < rbf.size(); x++)
            hiddenEval[x] = activationFunction(data, rbf.get(x), beta.get(x));
        for(int y = 0; y < weights.size(); y++){
            double out = 0;
            for(int z = 0; z < weights.get(y).length; z++){
                if(z==0)
                    out += weights.get(y)[z];
                else
                    out += weights.get(y)[z] * hiddenEval[z-1];
            }
            o[y] = out;
        }
        return o;
    }
    public static void sortData(ArrayList<double[]> d){
        inputs = new ArrayList<double[]>();
        outputs = new ArrayList<double[]>();
        for(double[] data : d){
            inputs.add(Arrays.copyOfRange(data, 0, input));
            outputs.add(Arrays.copyOfRange(data, input, data.length));
        }
    }
    public static void createNeuralNetwork(){
        createRBFNeurons();
        createOutputWeights();
    }
    public static double calculateEuclidDistance(double[] a, double[] b){
        double sum = 0;
        for(int x = 0; x < a.length; x++)
            sum += Math.pow(a[x]-b[x], 2);
        return Math.pow(sum, 0.5);
    }
    public static void createRBFNeurons(){

        rbf = new ArrayList<double[]>();
        beta = new ArrayList<Double>();

        //k means clustering
        int k = inputs.size()/2;
        ArrayList<double[]> kmeans = new ArrayList<double[]>();
        for(int kCount = 0; kCount < k; kCount++){
            double[] kmean = new double[input];
            for(int x = 0; x < kmean.length; x++)
                kmean[x] = Math.random();
            kmeans.add(kmean);
            beta.add(1.0); //can be anything, because edited later
        }

        for(int q = 0; q < 10000; q++) {

            Map<Integer, ArrayList<Integer>> indexStorage = new HashMap<Integer, ArrayList<Integer>>();

            for (int dataIter = 0; dataIter < inputs.size(); dataIter++) {
                double sDist = 999;
                int sIndex = 0;
                for (int kMeansIter = 0; kMeansIter < kmeans.size(); kMeansIter++) {
                    double calc = calculateEuclidDistance(inputs.get(dataIter), kmeans.get(kMeansIter));
                    if (calc < sDist) {
                        sDist = calc;
                        sIndex = kMeansIter;
                    }
                }
                if (indexStorage.get(sIndex) == null) {
                    ArrayList<Integer> a = new ArrayList<Integer>();
                    a.add(dataIter);
                    indexStorage.put(sIndex, a);
                } else
                    indexStorage.get(sIndex).add(dataIter);
            }
            for (int key : indexStorage.keySet()) {
                double[] average = new double[input];
                int counter = 0;
                double betaSum = 0;
                for (int index : indexStorage.get(key)) {
                    for (int x = 0; x < inputs.get(index).length; x++)
                        average[x] += inputs.get(index)[x];
                    betaSum += calculateEuclidDistance(inputs.get(index), kmeans.get(key));
                    counter += 1;
                }
                beta.set(key, betaSum/counter);
                for (int y = 0; y < average.length; y++)
                    average[y] = average[y] / counter;
                kmeans.set(key, average);
            }
        }
        rbf = kmeans;

    }
    public static void createOutputWeights(){
        weights = new ArrayList<double[]>();
        int numberOfHidden = rbf.size();
        double bias = Math.random();
        for(int numOut = 0; numOut < output; numOut++){
            double[] weight = new double[numberOfHidden+1];
            weight[0] = bias;
            for(int numIn = 1; numIn < weight.length; numIn++)
                weight[numIn] = Math.random();
            weights.add(weight);
        }
    }
}
