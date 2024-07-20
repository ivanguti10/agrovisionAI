import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomeComponent } from './components/home/home.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
//Angular Material

import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatChipsModule } from '@angular/material/chips';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatDialogModule } from '@angular/material/dialog';
import { MatDividerModule } from '@angular/material/divider';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatRadioModule } from '@angular/material/radio';
import { MatSelectModule } from '@angular/material/select';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSliderModule } from '@angular/material/slider';
import { MatSnackBarModule } from '@angular/material/snack-bar';
import { MatTableModule } from '@angular/material/table';
import { MatTabsModule } from '@angular/material/tabs';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { ReactiveFormsModule } from '@angular/forms';  // Asegúrate de importar esto



import { DialogImageComponent } from './components/shared/dialog-image/dialog-image.component';
import { DialogContentComponent } from './components/shared/dialog-content/dialog-content.component';

import { HttpClientModule } from '@angular/common/http';
import { YucaComponent } from './components/modelos/yuca/yuca.component';
import { YucaEjemplosComponent } from './components/modelos/yuca_ejemplos/yuca_ejemplos.component';
import { YucaModeloComponent } from './components/modelos/yuca-modelo/yuca-modelo.component';
import { YucaDuplicadosComponent } from './components/modelos/yuca-duplicados/yuca-duplicados.component';
import { YucaClasificarComponent } from './components/modelos/yuca-clasificar/yuca-clasificar.component';
import { ConteoComponent } from './components/conteo/conteo/conteo.component';




@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    DialogContentComponent,
    DialogImageComponent,
    YucaComponent,
    DialogImageComponent,
    YucaEjemplosComponent,
    YucaModeloComponent,
    YucaDuplicadosComponent,
    YucaClasificarComponent,
    ConteoComponent,
    ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    BrowserAnimationsModule,
    MatButtonModule,
    MatCardModule,
    MatCheckboxModule,
    MatChipsModule,
    MatDatepickerModule,
    MatButtonToggleModule,
    MatDialogModule,
    MatDividerModule,
    MatExpansionModule,
    MatGridListModule,
    MatIconModule,
    MatInputModule,
    MatListModule,
    MatMenuModule,
    MatPaginatorModule,
    MatProgressSpinnerModule,
    MatRadioModule,
    MatSelectModule,
    MatSidenavModule,
    MatSlideToggleModule,
    MatSliderModule,
    MatSnackBarModule,
    MatTableModule,
    MatTabsModule,
    MatToolbarModule,
    MatTooltipModule,
    ReactiveFormsModule  // Asegúrate de incluir esto en los imports

  ],

  providers: [],
  bootstrap: [AppComponent]
  

})
export class AppModule { }
