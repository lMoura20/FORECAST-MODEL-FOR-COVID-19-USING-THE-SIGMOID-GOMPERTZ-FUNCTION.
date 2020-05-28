# FORECAST-MODEL-FOR-COVID-19-USING-THE-SIGMOID-GOMPERTZ-FUNCTION.
This code was created with the function of helping in the effort to fight the new corona virus.

This study has some work premises which are shown below:**

- It is considered that the use of data on the number of infected persons does not correspond to reality, since it is evidenced the inability to carry out tests on the entire population and consequently to acquire data from lethality, etc;


- The data provided on the number of deaths are considered more accurate although it is not possible to differentiate between deaths caused by other respiratory syndromes or in cases where Covid is not the main cause of death;


- The data collected for this work were made available by: https://coronavirus.jhu.edu/map.html; 


- The data analyzed were performed based on the following countries: US; UK; ITALY; SPAIN; FRANCE; BRAZIL; BELGIUM; GERMANY; CHINA;


- It is necessary to consider that each country has a strategy to combat proliferation, which causes variability in growth rates, maximum value, etc. Individual protection measures collections, educational measures, mass tests and conditions of the public health system are some of the points to be considered in the data evaluation;


- The daily variability of the data is due to the way in which each country updates the cases, making it possible to perceive a high variability of the data mainly on weekends. To address this problem, the moving average technique was used to smooth this phenomenon;


- It is necessary to note that due to the time of incubation and / or evolution of the Coronavirus in the organism, to observe the data in a 'window' around a period for the evaluation of the dynamics of the phenomenon of governmental actions;


- The calculations performed must be considered within a history of government actions and therefore events such as lockdown can provide a significant change in the behavior of the curve;


- The projection of cases of death can be considered as an inference of the number of people who are infected;


- Considering that many countries registered the first cases before Brazil, this work seeks to evaluate the applicability of the model in the forecast of cases in Brazil. An important point in this assessment is to present the dynamics that the point of greatest daily record of cases with the consequent drop in the number of cases indicates a dynamic characteristic of this pandemic.


- It must be considered that the reality of the data demonstrates a social dynamic that directly impacts the divergences of the model. An example is to observe that the number of cases tends to present some steps when the number of cases has significantly reduced, which could be an indication of the economic opening measures.


