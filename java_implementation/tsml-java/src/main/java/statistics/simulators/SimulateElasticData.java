/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 

package statistics.simulators;

import weka.core.Instances;

/**
 *
 * @author ajb



*/
public class SimulateElasticData {
    static DataSimulator sim;
    static double warpPercent=0.1;
    public static Instances generateElasticData(int seriesLength, int []casesPerClass)
    {
        ElasticModel[] elastic = new ElasticModel[casesPerClass.length];
        populateElasticModels(elastic,seriesLength);
        sim = new DataSimulator(elastic);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;
        
    }        
//Stop it being a step
    private static void populateElasticModels(ElasticModel[] m, int seriesLength){
        if(m.length!=2)
            System.out.println("ONLY IMPLEMENTED FOR TWO CLASSES");
//Create two models with same interval but different shape. 
        ElasticModel m1=new ElasticModel();
        m1.setSeriesLength(seriesLength);
        ElasticModel m2=new ElasticModel();
        m2.setSeriesLength(seriesLength);
//Dont use  sine, its too easy        
        DictionaryModel.ShapeType[] vals={
            DictionaryModel.ShapeType.TRIANGLE,
DictionaryModel.ShapeType.STEP,DictionaryModel.ShapeType.HEADSHOULDERS         
        };
//DictionaryModel.ShapeType.SPIKE,        
        DictionaryModel.ShapeType shape=vals[Model.rand.nextInt(vals.length)];
        m1.setShape(shape);
        shape=vals[Model.rand.nextInt(vals.length)];
//Choose a shape nt equal to m1 
        while(shape==m1.getShape())
            shape=vals[Model.rand.nextInt(vals.length)];
        m2.setShape(shape);
        m[0]=m1;
        m[1]=m2;
        
    }
}
