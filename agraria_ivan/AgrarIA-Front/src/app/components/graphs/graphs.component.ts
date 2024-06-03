import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import Swal from 'sweetalert2';
import { FormControl } from '@angular/forms';
import { MatDialog, MatDialogConfig } from '@angular/material/dialog';
import { DialogContentComponent } from '../shared/dialog-content/dialog-content.component';

@Component({
  selector: 'app-graphs',
  templateUrl: './graphs.component.html',
  styleUrls: ['./graphs.component.scss']
})
export class GraphsComponent implements OnInit {
  @ViewChild('archivoInput') archivoInput!: ElementRef; // Referencia al input de archivos
  @ViewChild('spinner') spinner!: ElementRef; // Referencia al spinner
  @ViewChild('subiryprocesarButton') subiryprocesarButton!: ElementRef; // Referencia al botón "Subir y procesar"
  ngOnInit(): void {
  }
  loading = false;
  graficos: any; // Aquí almacenaremos los gráficos obtenidos
  archivosSeleccionados: boolean = false;
  archivos: any;
  disableSelect = new FormControl(false);

  constructor(private http: HttpClient, public dialog: MatDialog) { }

  getData(event: any) {
    this.archivos = event.target.files
  }

  openImageDialog(imageSrc: any) {
    const imagenSeleccionada = imageSrc;
    const dialogConfig = new MatDialogConfig();
    dialogConfig.data = { imageSrc: imagenSeleccionada };
    dialogConfig.height = '80%';
    this.dialog.open(DialogContentComponent, dialogConfig);
  }
  visualizarAtributos() {
    this.loading = true

    if (this.archivos.length === 0) {
      this.archivosSeleccionados = false; // No se han seleccionado archivos
      return;
    }

    this.archivosSeleccionados = true; // Se han seleccionado archivos

    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    this.http.post<any>('http://127.0.0.1:5000/visualizar_atributos', formData).subscribe((response) => {
      this.graficos = response;
      this.loading = false;
      Swal.fire({
        position: 'top-end',
        icon: 'success',
        title: 'Se han generado los gráficos',
        showConfirmButton: false,
        timer: 2000,
        backdrop: false
      })
    });
  }

  analisisDeInformacion() {
    this.loading = true
    if (this.archivos.length === 0) {
      this.archivosSeleccionados = false; // No se han seleccionado archivos
      return;
    }

    this.archivosSeleccionados = true; // Se han seleccionado archivos

    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    this.http.post<any>('http://127.0.0.1:5000/analisis_informacion', formData).subscribe((response) => {
      this.graficos = response;
      this.loading = false;
      Swal.fire({
        position: 'top-end',
        icon: 'success',
        title: 'Se han generado los gráficos',
        showConfirmButton: false,
        timer: 2000,
        backdrop: false
      })
    });
  }
  algorithmValue: any;
  machineLearning() {
    this.loading = true
    if (this.archivos.length === 0) {
      this.archivosSeleccionados = false; // No se han seleccionado archivos
      return;
    }

    this.archivosSeleccionados = true; // Se han seleccionado archivos

    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }
    // Agrega el valor seleccionado en el 'select' al FormData


    formData.append('algoritmo', this.algorithmValue);

    this.http.post<any>('http://127.0.0.1:5000/machine_learning', formData).subscribe((response) => {
      this.graficos = response;
      this.loading = false;
      Swal.fire({
        position: 'top-end',
        icon: 'success',
        title: 'Se han generado los gráficos',
        showConfirmButton: false,
        timer: 2000,
        backdrop: false,
      })
    });
  }
  changeSelect(event: any) {
    this.algorithmValue = event;

  }

}
