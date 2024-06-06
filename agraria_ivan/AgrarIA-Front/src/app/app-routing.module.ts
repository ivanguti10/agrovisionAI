import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { GraphsComponent } from './components/graphs/graphs.component';
import { XaiComponent } from './components/modelos/xai/xai.component';
import { RnnComponent } from './components/modelos/rnn/rnn.component';
import { YucaComponent } from './components/modelos/yuca/yuca.component';
import { YucaEjemplosComponent } from './components/modelos/yuca_ejemplos/yuca_ejemplos.component';





const routes: Routes = [
  { path: 'home', component: HomeComponent },
  { path: 'graphs', component: GraphsComponent },
  { path: 'xai', component: XaiComponent },
  { path: 'rnn', component: RnnComponent },
  { path: 'yuca', component: YucaComponent },
  { path: 'yuca_ejemplos', component: YucaEjemplosComponent },
  { path: '', redirectTo: '/yuca_ejemplos', pathMatch: 'full' }, // Redirigir a ComponentA por defecto
  { path: '**', redirectTo: 'home' },
];
@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
