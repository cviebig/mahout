/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.sparkbindings.drm.classification

import org.apache.mahout.sparkbindings.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings.drm._
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel
import org.apache.mahout.classifier.naivebayes.training.{StandardThetaTrainer, ComplementaryThetaTrainer}


object NaiveBayesRunner {
  //TODO add
}

class TrainNaiveBayes[K](val alphaI: Float = 1.0, trainComplementary: Boolean = false) {

  def createThetaTrainer(weightsPerFeature: Vector, weightsPerLabel: Vector) = {
    if (trainComplementary) {
      new ComplementaryThetaTrainer(weightsPerFeature, weightsPerLabel, alphaI)
    } else {
      new StandardThetaTrainer(weightsPerFeature, weightsPerLabel, alphaI)
    }
  }

  def run(observationsPerClass: DrmLike[K]*): Unit = {

    val weightsPerLabelAndFeature = dense(observationsPerClass.map(_.colSums))
    val weightsPerFeature = weightsPerLabelAndFeature.colSums
    val weightsPerLabel = weightsPerLabelAndFeature.rowSums

    val thetaTrainer = createThetaTrainer(weightsPerFeature, weightsPerLabel)

    for (labelIndex <- 0 until weightsPerLabelAndFeature.nrow) {
      thetaTrainer.train(labelIndex, weightsPerLabelAndFeature(labelIndex, ::))
    }

    new NaiveBayesModel(weightsPerLabelAndFeature, weightsPerFeature, weightsPerLabel,
                        thetaTrainer.retrievePerLabelThetaNormalizer(), alphaI)
  }
}
