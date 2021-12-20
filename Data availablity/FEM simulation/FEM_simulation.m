function out = FEM_simulation
%
% Model.m
%
% Model exported on Jul 4 2020, 11:16 by COMSOL 5.5.0.359.

import com.comsol.model.*
import com.comsol.model.util.*

load('G:\structure\unit_cell_data.mat','unit_cell_data')

for i=1:length(unit_cell_data)

unit_cell = find(flipud(unit_cell_data{1,i})==0);
    
model = ModelUtil.create('Model');

model.modelPath('G:\2020\Test');

model.param.set('l', '0.001');
model.param.set('n', '50');
model.param.set('a', 'n*l');
model.param.set('k1', '0');
model.param.set('kx', 'k1*pi/a');
model.param.set('k2', '0');
model.param.set('ky', 'k2*pi/a');
model.param.set('ela', '1.7e9');
model.param.set('poi', '0.4');
model.param.set('den', '1150');
model.param.set('ct', 'sqrt(ela/2/(1+poi)/den)');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 2);

model.component('comp1').mesh.create('mesh1');

model.component('comp1').geom('geom1').useConstrDim(false);
model.component('comp1').geom('geom1').create('sq1', 'Square');
model.component('comp1').geom('geom1').feature('sq1').set('size', 'l');
model.component('comp1').geom('geom1').create('arr1', 'Array');
model.component('comp1').geom('geom1').feature('arr1').set('fullsize', {'n' 'n'});
model.component('comp1').geom('geom1').feature('arr1').set('displ', {'l' 'l'});
model.component('comp1').geom('geom1').feature('arr1').selection('input').set({'sq1'});
model.component('comp1').geom('geom1').run;
model.component('comp1').geom('geom1').run('fin');

model.component('comp1').common.create('mpf1', 'ParticipationFactors');

model.component('comp1').physics.create('solid', 'SolidMechanics', 'geom1');
model.component('comp1').physics('solid').selection.set(unit_cell);
model.component('comp1').physics('solid').create('pc1', 'PeriodicCondition', 1);
model.component('comp1').physics('solid').feature('pc1').selection.set([1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95 97 99 5051 5052 5053 5054 5055 5056 5057 5058 5059 5060 5061 5062 5063 5064 5065 5066 5067 5068 5069 5070 5071 5072 5073 5074 5075 5076 5077 5078 5079 5080 5081 5082 5083 5084 5085 5086 5087 5088 5089 5090 5091 5092 5093 5094 5095 5096 5097 5098 5099 5100]);
model.component('comp1').physics('solid').create('pc2', 'PeriodicCondition', 1);
model.component('comp1').physics('solid').feature('pc2').selection.set([2 101 103 202 204 303 305 404 406 505 507 606 608 707 709 808 810 909 911 1010 1012 1111 1113 1212 1214 1313 1315 1414 1416 1515 1517 1616 1618 1717 1719 1818 1820 1919 1921 2020 2022 2121 2123 2222 2224 2323 2325 2424 2426 2525 2527 2626 2628 2727 2729 2828 2830 2929 2931 3030 3032 3131 3133 3232 3234 3333 3335 3434 3436 3535 3537 3636 3638 3737 3739 3838 3840 3939 3941 4040 4042 4141 4143 4242 4244 4343 4345 4444 4446 4545 4547 4646 4648 4747 4749 4848 4850 4949 4951 5050]);

model.component('comp1').mesh('mesh1').create('fq1', 'FreeQuad');

model.component('comp1').view('view1').axis.set('xmin', -0.03089308924973011);
model.component('comp1').view('view1').axis.set('xmax', 0.08089308440685272);
model.component('comp1').view('view1').axis.set('ymin', -0.004877591505646706);
model.component('comp1').view('view1').axis.set('ymax', 0.054877594113349915);

model.component('comp1').physics('solid').feature('lemm1').set('E_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').set('E', 'ela');
model.component('comp1').physics('solid').feature('lemm1').set('nu_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').set('nu', 'poi');
model.component('comp1').physics('solid').feature('lemm1').set('rho_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').set('rho', 'den');
model.component('comp1').physics('solid').feature('pc1').set('PeriodicType', 'Floquet');
model.component('comp1').physics('solid').feature('pc1').set('kFloquet', {'kx'; 'ky'; '0'});
model.component('comp1').physics('solid').feature('pc2').set('PeriodicType', 'Floquet');
model.component('comp1').physics('solid').feature('pc2').set('kFloquet', {'kx'; 'ky'; '0'});

model.component('comp1').mesh('mesh1').run;

model.study.create('std1');
model.study('std1').create('param', 'Parametric');
model.study('std1').create('eig', 'Eigenfrequency');

model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').attach('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('e1', 'Eigenvalue');
model.sol.create('sol2');
model.sol('sol2').study('std1');
model.sol('sol2').label('Parametric Solutions 1');

model.batch.create('p1', 'Parametric');
model.batch('p1').create('so1', 'Solutionseq');
model.batch('p1').study('std1');

model.result.create('pg2', 'PlotGroup1D');
model.result('pg2').set('data', 'dset2');
model.result('pg2').create('glob1', 'Global');
model.result.export.create('plot1', 'Plot');

model.study('std1').feature('param').set('pname', {'k1' 'k2'});
model.study('std1').feature('param').set('plistarr', {'1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1 1 1 1 1 1 1 1 1 1' '1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0 0 0 0 0 0 0 0 0 0 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1'});
model.study('std1').feature('param').set('punit', {'' ''});
model.study('std1').feature('eig').set('neigs', 10);
model.study('std1').feature('eig').set('neigsactive', true);
model.study('std1').feature('eig').set('ngen', 5);

model.sol('sol1').attach('std1');
model.sol('sol1').feature('e1').set('transform', 'eigenfrequency');
model.sol('sol1').feature('e1').set('neigs', 10);
model.sol('sol1').feature('e1').set('shift', '1[Hz]');
model.sol('sol1').feature('e1').set('eigvfunscale', 'maximum');
model.sol('sol1').feature('e1').set('eigvfunscaleparam', 7.069999999999999E-8);
model.sol('sol1').feature('e1').feature('aDef').set('cachepattern', true);
model.sol('sol1').runAll;

model.batch('p1').set('control', 'param');
model.batch('p1').set('pname', {'k1' 'k2'});
model.batch('p1').set('plistarr', {'1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1 1 1 1 1 1 1 1 1 1' '1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0 0 0 0 0 0 0 0 0 0 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1'});
model.batch('p1').set('punit', {'' ''});
model.batch('p1').set('err', true);
model.batch('p1').feature('so1').set('seq', 'sol1');
model.batch('p1').feature('so1').set('psol', 'sol2');
model.batch('p1').feature('so1').set('param', {'"k1","1","k2","1"' '"k1","0.9","k2","0.9"' '"k1","0.8","k2","0.8"' '"k1","0.7","k2","0.7"' '"k1","0.6","k2","0.6"' '"k1","0.5","k2","0.5"' '"k1","0.4","k2","0.4"' '"k1","0.3","k2","0.3"' '"k1","0.2","k2","0.2"' '"k1","0.1","k2","0.1"'  ...
'"k1","0","k2","0"' '"k1","0.1","k2","0"' '"k1","0.2","k2","0"' '"k1","0.3","k2","0"' '"k1","0.4","k2","0"' '"k1","0.5","k2","0"' '"k1","0.6","k2","0"' '"k1","0.7","k2","0"' '"k1","0.8","k2","0"' '"k1","0.9","k2","0"'  ...
'"k1","1","k2","0"' '"k1","1","k2","0.1"' '"k1","1","k2","0.2"' '"k1","1","k2","0.3"' '"k1","1","k2","0.4"' '"k1","1","k2","0.5"' '"k1","1","k2","0.6"' '"k1","1","k2","0.7"' '"k1","1","k2","0.8"' '"k1","1","k2","0.9"'  ...
'"k1","1","k2","1"'});
model.batch('p1').attach('std1');
model.batch('p1').run;

model.result('pg2').set('xlabel', 'Solution number');
model.result('pg2').set('xlabelactive', false);
model.result('pg2').feature('glob1').set('expr', {'freq*a/ct'});
model.result('pg2').feature('glob1').set('unit', {'Hz'});
model.result('pg2').feature('glob1').set('descr', {''});
model.result('pg2').feature('glob1').set('const', {'solid.refpntx' '0' 'Reference point for moment computation, x coordinate'; 'solid.refpnty' '0' 'Reference point for moment computation, y coordinate'; 'solid.refpntz' '0' 'Reference point for moment computation, z coordinate'});
model.result('pg2').feature('glob1').set('xdatasolnumtype', 'outer');
model.result.export('plot1').set('filename', ['G:\dispersion relation\model' num2str(i) '.txt']);
model.result.export('plot1').run;

end
out = model;
