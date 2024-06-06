import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import Swal from 'sweetalert2';
import { MatDialog, MatDialogConfig } from '@angular/material/dialog';
import { DialogContentComponent } from '../../shared/dialog-content/dialog-content.component';
import { DomSanitizer } from '@angular/platform-browser';
import { DialogImageComponent } from '../../shared/dialog-image/dialog-image.component';


@Component({
  selector: 'app-yuca_ejemplos',
  templateUrl: './yuca_ejemplos.component.html',
  styleUrls: ['./yuca_ejemplos.component.scss']
})
export class YucaEjemplosComponent implements OnInit {
  @ViewChild('archivoInput') archivoInput!: ElementRef;
  archivos: any;
  loading = false;
  graficos: any = [];
  archivosSeleccionados = false;
  mostrarEjemplos = false;
  
  constructor(private http: HttpClient, public dialog: MatDialog, public sanitizer: DomSanitizer) {}


  ngOnInit(): void {}

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
  

  openHTMLDialog(imageBase64: string, text: string): void {
    this.dialog.open(DialogImageComponent, {
      data: { image: imageBase64, text: text }
    });
  }

  yuca(): void {
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

    this.http.post<any>('http://127.0.0.1:5000/yuca', formData).subscribe(
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

  yuca_CBB(): void {
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

    this.http.post<any>('http://127.0.0.1:5000/yuca_CBB', formData).subscribe(
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

  yuca_CGM(): void {
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

    this.http.post<any>('http://127.0.0.1:5000/yuca_CGM', formData).subscribe(
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

  yuca_CDM(): void {
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

    this.http.post<any>('http://127.0.0.1:5000/yuca_CDM', formData).subscribe(
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

  yuca_CBSD(): void {
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

    this.http.post<any>('http://127.0.0.1:5000/yuca_CBSD', formData).subscribe(
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
