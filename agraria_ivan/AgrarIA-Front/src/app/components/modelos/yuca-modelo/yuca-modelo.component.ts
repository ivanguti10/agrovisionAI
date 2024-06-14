import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { MatDialog, MatDialogConfig } from '@angular/material/dialog';
import { DomSanitizer } from '@angular/platform-browser';
import Swal from 'sweetalert2';
import { DialogImageComponent } from '../../shared/dialog-image/dialog-image.component';

@Component({
  selector: 'app-yuca-modelo',
  templateUrl: './yuca-modelo.component.html',
  styleUrls: ['./yuca-modelo.component.scss']
})
export class YucaModeloComponent {

  archivos: any;
  loading = false;
  graficos: any = [];
  archivosSeleccionados = false;
  mostrarEjemplos = false;

  constructor(private http: HttpClient, public dialog: MatDialog, public sanitizer: DomSanitizer) {}
  
  getData(event: any) {
    this.archivos = event.target.files;
  }

  openImageDialog(imageBase64: string): void {
    const dialogConfig = new MatDialogConfig();
    dialogConfig.data = {
      imageSrc: imageBase64,
    };
    dialogConfig.hasBackdrop = true;
    dialogConfig.maxHeight = '100%';
    dialogConfig.maxWidth = '100%';
    dialogConfig.minHeight = 'auto';
    dialogConfig.minWidth = 'auto';
    this.dialog.open(DialogImageComponent, dialogConfig);
  }

  cargarModelo(): void {
    this.loading = true;
    this.mostrarEjemplos = false;

    if (!this.archivos || this.archivos.length === 0) {
      this.archivosSeleccionados = false;
      this.loading = false;
      return;
    }

    this.archivosSeleccionados = true;
    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    this.http.post<any>('http://127.0.0.1:5000/cargarModelo', formData).subscribe(
      (response) => {
        // Este bloque se ejecuta si el modelo se carga correctamente
        this.graficos = response;
        this.loading = false;
        this.mostrarEjemplos = true;
        Swal.fire({
          position: 'top-end',
          icon: 'success',
          title: 'Se han cargado las características del modelo',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      },
      (error) => {
        // Este bloque se ejecuta en caso de error
        this.loading = false;
        Swal.fire({
          position: 'top-end',
          icon: 'error', // Cambiado a 'error' para reflejar el fallo
          title: 'Error al cargar el modelo',
          text: 'Por favor, inténtalo de nuevo.', // Mensaje adicional opcional
          showConfirmButton: true,
          timer: 5000, // Ajusta el tiempo según necesites
          backdrop: true
        });
      }
    );
  }

  cluster1(): void {
    this.loading = true;
    this.mostrarEjemplos = false;

    if (!this.archivos || this.archivos.length === 0) {
      this.archivosSeleccionados = false;
      this.loading = false;
      return;
    }

    this.archivosSeleccionados = true;
    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    this.http.post<any>('http://127.0.0.1:5000/cluster1', formData).subscribe(
      (response) => {
        this.graficos = response;
        this.loading = false;
        this.mostrarEjemplos = true;
        Swal.fire({
          position: 'top-end',
          icon: 'success',
          title: 'Se han generado los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      },
      (error) => {
        this.loading = false;
        Swal.fire({
          position: 'top-end',
          icon: 'error',
          title: 'Error al generar los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      }
    );
  }


  cluster2(): void {
    this.loading = true;
    this.mostrarEjemplos = false;

    if (!this.archivos || this.archivos.length === 0) {
      this.archivosSeleccionados = false;
      this.loading = false;
      return;
    }

    this.archivosSeleccionados = true;
    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    this.http.post<any>('http://127.0.0.1:5000/cluster2', formData).subscribe(
      (response) => {
        this.graficos = response;
        this.loading = false;
        this.mostrarEjemplos = true;
        Swal.fire({
          position: 'top-end',
          icon: 'success',
          title: 'Se han generado los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      },
      (error) => {
        this.loading = false;
        Swal.fire({
          position: 'top-end',
          icon: 'error',
          title: 'Error al generar los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      }
    );
  }

  cluster3(): void {
    this.loading = true;
    this.mostrarEjemplos = false;

    if (!this.archivos || this.archivos.length === 0) {
      this.archivosSeleccionados = false;
      this.loading = false;
      return;
    }

    this.archivosSeleccionados = true;
    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    this.http.post<any>('http://127.0.0.1:5000/cluster3', formData).subscribe(
      (response) => {
        this.graficos = response;
        this.loading = false;
        this.mostrarEjemplos = true;
        Swal.fire({
          position: 'top-end',
          icon: 'success',
          title: 'Se han generado los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      },
      (error) => {
        this.loading = false;
        Swal.fire({
          position: 'top-end',
          icon: 'error',
          title: 'Error al generar los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      }
    );
  }

  cluster4(): void {
    this.loading = true;
    this.mostrarEjemplos = false;

    if (!this.archivos || this.archivos.length === 0) {
      this.archivosSeleccionados = false;
      this.loading = false;
      return;
    }

    this.archivosSeleccionados = true;
    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    this.http.post<any>('http://127.0.0.1:5000/cluster4', formData).subscribe(
      (response) => {
        this.graficos = response;
        this.loading = false;
        this.mostrarEjemplos = true;
        Swal.fire({
          position: 'top-end',
          icon: 'success',
          title: 'Se han generado los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      },
      (error) => {
        this.loading = false;
        Swal.fire({
          position: 'top-end',
          icon: 'error',
          title: 'Error al generar los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      }
    );
  }

  cluster5(): void {
    this.loading = true;
    this.mostrarEjemplos = false;

    if (!this.archivos || this.archivos.length === 0) {
      this.archivosSeleccionados = false;
      this.loading = false;
      return;
    }

    this.archivosSeleccionados = true;
    const formData = new FormData();
    for (let i = 0; i < this.archivos.length; i++) {
      formData.append('archivo', this.archivos[i]);
    }

    this.http.post<any>('http://127.0.0.1:5000/cluster5', formData).subscribe(
      (response) => {
        this.graficos = response;
        this.loading = false;
        this.mostrarEjemplos = true;
        Swal.fire({
          position: 'top-end',
          icon: 'success',
          title: 'Se han generado los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      },
      (error) => {
        this.loading = false;
        Swal.fire({
          position: 'top-end',
          icon: 'error',
          title: 'Error al generar los gráficos',
          showConfirmButton: false,
          timer: 2000,
          backdrop: false
        });
      }
    );
  }

}
