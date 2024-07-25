import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private apiUrl = 'http://localhost:5000/predict';  // Asegúrate de que esta URL coincida con la de tu backend

  constructor(private http: HttpClient) { }

  predict(data: any): Observable<any> {
    return this.http.post<any>(this.apiUrl, data);
  }
}
