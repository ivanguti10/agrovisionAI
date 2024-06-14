import { ComponentFixture, TestBed } from '@angular/core/testing';

import { YucaModeloComponent } from './yuca-modelo.component';

describe('YucaModeloComponent', () => {
  let component: YucaModeloComponent;
  let fixture: ComponentFixture<YucaModeloComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [YucaModeloComponent]
    });
    fixture = TestBed.createComponent(YucaModeloComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
