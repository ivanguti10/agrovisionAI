import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { YucaComponent } from './components/modelos/yuca/yuca.component';
import { YucaEjemplosComponent } from './components/modelos/yuca_ejemplos/yuca_ejemplos.component';
import { YucaModeloComponent } from './components/modelos/yuca-modelo/yuca-modelo.component';
import { YucaDuplicadosComponent } from './components/modelos/yuca-duplicados/yuca-duplicados.component';
import { YucaClasificarComponent } from './components/modelos/yuca-clasificar/yuca-clasificar.component';
import { ConteoComponent } from './components/conteo/conteo/conteo.component';


const routes: Routes = [
  { path: 'home', component: HomeComponent },
  { path: 'yuca', component: YucaComponent },
  { path: 'yuca_ejemplos', component: YucaEjemplosComponent },
  { path: 'yuca-modelo', component: YucaModeloComponent },
  { path: 'yuca-duplicados', component: YucaDuplicadosComponent },
  { path: 'yuca-clasificar', component: YucaClasificarComponent },
  { path: 'conteo', component: ConteoComponent },
  { path: '**', redirectTo: 'home' },
];
@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
