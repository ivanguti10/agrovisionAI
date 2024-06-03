import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import Swal from 'sweetalert2';
import { FormControl } from '@angular/forms';
import { MatDialog, MatDialogConfig } from '@angular/material/dialog';
import { DialogContentComponent } from '../../shared/dialog-content/dialog-content.component';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { timeout } from 'rxjs';
import { DialogImageComponent } from '../../shared/dialog-image/dialog-image.component';
import { catchError, throwError } from 'rxjs';

@Component({
  selector: 'app-rnn',
  templateUrl: './rnn.component.html',
  styleUrls: ['./rnn.component.scss']
})
export class RnnComponent {
  @ViewChild('archivoInput') archivoInput!: ElementRef; // Referencia al input de archivos
  @ViewChild('spinner') spinner!: ElementRef; // Referencia al spinner
  @ViewChild('subiryprocesarButton') subiryprocesarButton!: ElementRef; // Referencia al botón "Subir y procesar"
  ngOnInit(): void {
  }
  loading = false;
  ifGraficos = false;
  graficos: any; // Aquí almacenaremos los gráficos obtenidos
  archivosSeleccionados: boolean = false;
  archivos: any;
  archivos1: any;
  disableSelect = new FormControl(false);

  constructor(private http: HttpClient, public dialog: MatDialog, public sanitizer: DomSanitizer) { }

  getData(event: any) {
    this.archivos = event.target.files
  }

  openImageDialog(grafico: any) {
    const dialogConfig = new MatDialogConfig();
    dialogConfig.height = 'auto';
    dialogConfig.width = '100%';
    dialogConfig.data = { imageSrc: grafico.base64, imagenNormal: grafico.base64_2, imageCombined: grafico.combined_base64, text: grafico.text };
    this.dialog.open(DialogImageComponent, dialogConfig);
  }
  openHTMLDialog(grafico: any) {
    const dialogConfig = new MatDialogConfig();
    dialogConfig.data = { html: grafico.base64, text: grafico.text };
    dialogConfig.height = 'auto';
    dialogConfig.width = '100%';
    this.dialog.open(DialogContentComponent, dialogConfig);
  }

  lime() {
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

    this.http.post<any>('http://127.0.0.1:5000/lime', formData).pipe(
      catchError((error) => {
        console.error('Se produjo un error:', error);
        return throwError(error);
      })
    ).subscribe((response) => {
      this.graficos = response;
      this.loading = false;
      this.ifGraficos = true;
      Swal.fire({ position: 'top-end', icon: 'success', title: 'Se han generado los gráficos', showConfirmButton: false, timer: 2000, backdrop: false });
    }, (error) => {
      this.loading = false;
      Swal.fire({ icon: "error", text: "Ha habido un error al generar los gráficos.", footer: '<a href="/rnn">Limpiar archivo.</a>' });
    });
  }
  shap() {
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

    this.http.post<any>('http://127.0.0.1:5000/shap', formData).pipe(
      catchError((error) => {
        console.error('Se produjo un error:', error);
        return throwError(error);
      })
    ).subscribe((response) => {
      this.graficos = response;
      this.loading = false;
      this.ifGraficos = true;
      Swal.fire({ position: 'top-end', icon: 'success', title: 'Se han generado los gráficos', showConfirmButton: false, timer: 2000, backdrop: false });
    }, (error) => {
      this.loading = false;
      Swal.fire({ icon: "error", text: "Ha habido un error al generar los gráficos.", footer: '<a href="/rnn">Limpiar archivo.</a>' });
    });
  }

  resumenes() {
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

    this.http.post<any>('http://127.0.0.1:5000/resumenes', formData).pipe(
      catchError((error) => {
        console.error('Se produjo un error:', error);
        return throwError(error);
      })
    ).subscribe((response) => {
      this.graficos = response;
      this.loading = false;
      Swal.fire({ position: 'top-end', icon: 'success', title: 'Se han generado los gráficos', showConfirmButton: false, timer: 2000, backdrop: false });
    }, (error) => {
      this.loading = false;
      Swal.fire({ icon: "error", text: "Ha habido un error al generar los gráficos.", footer: '<a href="/rnn">Limpiar archivo.</a>' });
    });
  }

  algorithmValue: any;
  pytorch_conv() {
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

    this.http.post<any>('http://127.0.0.1:5000/pytorch_conv', formData).pipe(
      catchError((error) => {
        console.error('Se produjo un error:', error);
        return throwError(error);
      })
    ).subscribe((response) => {
      this.graficos = response;
      this.loading = false;
      Swal.fire({ position: 'top-end', icon: 'success', title: 'Se han generado los gráficos', showConfirmButton: false, timer: 2000, backdrop: false });
    }, (error) => {
      this.loading = false;
      Swal.fire({ icon: "error", text: "Ha habido un error al generar los gráficos.", footer: '<a href="/rnn">Limpiar archivo.</a>' });
    });
  }

  pytorch_conv2() {
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

    this.http.post<any>('http://127.0.0.1:5000/pytorch_conv2', formData).pipe(
      catchError((error) => {
        console.error('Se produjo un error:', error);
        return throwError(error);
      })
    ).subscribe((response) => {
      this.graficos = response;
      this.loading = false;
      Swal.fire({ position: 'top-end', icon: 'success', title: 'Se han generado los gráficos', showConfirmButton: false, timer: 2000, backdrop: false });
    }, (error) => {
      this.loading = false;
      Swal.fire({ icon: "error", text: "Ha habido un error al generar los gráficos.", footer: '<a href="/rnn">Limpiar archivo.</a>' });
    });
  }
  changeSelect(event: any) {
    this.algorithmValue = event;
  }
  python_conv_lime() {
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

    this.http.post<any>('http://127.0.0.1:5000/python_conv_lime', formData).pipe(
      catchError((error) => {
        console.error('Se produjo un error:', error);
        return throwError(error);
      })
    ).subscribe((response) => {
      this.graficos = response;
      const image_normal = this.graficos[0].base64;
      this.graficos[1].base64_2 = image_normal;
      this.graficos[2].base64_2 = image_normal;
      this.loading = false;
      Swal.fire({ position: 'top-end', icon: 'success', title: 'Se han generado los gráficos', showConfirmButton: false, timer: 2000, backdrop: false });
    }, (error) => {
      this.loading = false;
      Swal.fire({ icon: "error", text: "Ha habido un error al generar los gráficos.", footer: '<a href="/rnn">Limpiar archivo.</a>' });
    });
  }


}
