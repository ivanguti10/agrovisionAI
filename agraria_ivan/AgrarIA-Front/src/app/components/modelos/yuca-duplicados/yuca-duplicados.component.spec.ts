import { ComponentFixture, TestBed } from '@angular/core/testing';

import { YucaDuplicadosComponent } from './yuca-duplicados.component';

describe('YucaDuplicadosComponent', () => {
  let component: YucaDuplicadosComponent;
  let fixture: ComponentFixture<YucaDuplicadosComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [YucaDuplicadosComponent]
    });
    fixture = TestBed.createComponent(YucaDuplicadosComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
