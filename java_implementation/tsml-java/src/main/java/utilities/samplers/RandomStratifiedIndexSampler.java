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
 
package utilities.samplers;

import utilities.ArrayUtilities;
import weka.core.Instances;

import java.util.List;
import java.util.Random;

import static utilities.InstanceTools.*;
import static utilities.Utilities.argMax;

public class RandomStratifiedIndexSampler implements Sampler{

    private List<List<Integer>> instancesByClass;
    private double[] classDistribution;
    private double[] classSamplingProbabilities;
    private int count;
    private Random random;
    private int maxCount;

    public RandomStratifiedIndexSampler(Random random){
        this.random = random;
    }

    public RandomStratifiedIndexSampler(){
        random = new Random();
    }

    public void setInstances(Instances instances) {
        instancesByClass = indexByClass(instances);
        classDistribution = classDistribution(instances);
        classSamplingProbabilities = classDistribution(instances);
        count = 0;
        maxCount = instances.size();
    }

    public boolean hasNext() {
        return count < maxCount;
    }

    public Integer next() {
        int sampleClass = argMax(classSamplingProbabilities, random);
        List<Integer> homogeneousInstances = instancesByClass.get(sampleClass); // instances of the class value
        int sampledInstance = homogeneousInstances.remove(random.nextInt(homogeneousInstances.size()));
        classSamplingProbabilities[sampleClass]--;
        ArrayUtilities.add(classSamplingProbabilities, classDistribution);
        return sampledInstance;
    }
}
